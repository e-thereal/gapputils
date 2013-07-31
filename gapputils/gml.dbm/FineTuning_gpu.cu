/*
 * FineTuning_gpu.cu
 *
 *  Created on: Jul 16, 2013
 *      Author: tombr
 */

#include "FineTuning.h"

#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/random.hpp>
#include <tbblas/mask.hpp>
#include <tbblas/io.hpp>
#include <tbblas/linalg.hpp>

#include <omp.h>

#include "math.hpp"

namespace gml {

namespace dbm {

FineTuningChecker::FineTuningChecker() {
  FineTuning test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(Dataset, test);
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(GpuCount, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(LearningRateCL, test);
  CHECK_MEMORY_LAYOUT2(LearningRateRL, test);
  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(MeanFieldIterations, test);
  CHECK_MEMORY_LAYOUT2(InitialGibbsIterations, test);
  CHECK_MEMORY_LAYOUT2(GibbsIterations, test);
  CHECK_MEMORY_LAYOUT2(SampleCount, test);
  CHECK_MEMORY_LAYOUT2(OutputModel, test);
}

void FineTuning::update(IProgressMonitor* monitor) const {
  // Fine tuning is a combination of inference, sampling and training
  // I use mean field inference to infer the positive statistics (data-dependent)
  // Use sampling chains to infer the negative statistics (data-independent)
  // Use weight update rules from training to update the filters and bias terms

  using namespace tbblas;

  Logbook& dlog = getLogbook();

  const unsigned dimCount = Model::dimCount;
  typedef Model::value_t value_t;
  typedef complex<value_t> complex_t;
  typedef fft_plan<dimCount> plan_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor<complex_t, dimCount, true> ctensor_t;
  typedef tensor<complex_t, dimCount, false> host_ctensor_t;
  typedef std::vector<boost::shared_ptr<ctensor_t> > v_ctensor_t;
  typedef std::vector<boost::shared_ptr<v_ctensor_t> > vv_ctensor_t;
  typedef tensor_t::dim_t dim_t;

  typedef tensor<value_t, 2, true> matrix_t;

  // Get inputs
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getDataset();

  // Load model into device memory
  Model& dbm = *getInitialModel();

  // A DBM with 1 visible layer and n hidden layers has n layers for the sake of writing this code
  size_t cLayerCount = dbm.getWeights()->size();
  size_t rLayerCount = dbm.getWeightMatrices()->size();
  assert(cLayerCount && rLayerCount);

  dim_t visSize[cLayerCount], hidSize[cLayerCount], layerSize[cLayerCount];
  for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
    visSize[iLayer] = hidSize[iLayer] = layerSize[iLayer] = dbm.getHiddenBiases()->at(iLayer)->at(0)->size();
    visSize[iLayer][dimCount - 1] = dbm.getWeights()->at(iLayer)->at(0)->size()[dimCount - 1];
    hidSize[iLayer][dimCount - 1] = dbm.getWeights()->at(iLayer)->size();
  }

  dim_t rearrangeBlock[cLayerCount];
  rearrangeBlock[0] = inputs[0]->size() / visSize[0];
  rearrangeBlock[0][dimCount - 1] = 1;
  assert(rearrangeBlock[0] == dbm.getVisibleBlockSize());
  for (size_t iLayer = 1; iLayer < cLayerCount; ++iLayer)
    rearrangeBlock[iLayer] = dbm.getHiddenBiases()->at(iLayer - 1)->at(0)->size() / dbm.getHiddenBiases()->at(iLayer)->at(0)->size();

  const size_t batchSize = getBatchSize();
  const size_t batchCount = inputs.size() / batchSize;
  int gibbsIterations = getInitialGibbsIterations();

  // Initialize constants
  value_t epsilonw[cLayerCount], epsilonhb[cLayerCount], epsilonvb;
  value_t epsBatchCL, epsBatchRL;
  for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
    epsilonw[iLayer] = 1.0 / batchSize / sum(*dbm.getMasks()->at(iLayer)) / visSize[iLayer][dimCount - 1];
    epsilonhb[iLayer] = 1.0 / batchSize / hidSize[iLayer][dimCount - 1];
  }
  epsilonvb = 1.0 / batchSize / visSize[0][dimCount - 1];

  value_t initialmomentum = 0.5;
  value_t finalmomentum = 0.9;
  value_t momentum;
  size_t momentumEpochs = 10;

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  const int gpuCount = getGpuCount();

  if (deviceCount < gpuCount) {
    dlog(Severity::Warning) << "Only " << deviceCount << " CUDA-enabled devices found, where " << gpuCount << " are required according to GpuCount. Aborting!";
    return;
  }

  assert(omp_get_num_threads() == 1);

  cudaSetDevice(0);
  omp_set_dynamic(0);
  omp_set_num_threads(gpuCount);

  boost::shared_ptr<Model> outDbm(new Model());
  boost::shared_ptr<vv_host_tensor_t> outFilters(new vv_host_tensor_t(cLayerCount));
  boost::shared_ptr<vv_host_tensor_t> outC(new vv_host_tensor_t(cLayerCount));
  boost::shared_ptr<v_host_tensor_t> outMasks(new v_host_tensor_t(cLayerCount));
  boost::shared_ptr<v_host_matrix_t> outMatrices(new v_host_matrix_t(rLayerCount));
  boost::shared_ptr<v_host_matrix_t> outFlatBiases(new v_host_matrix_t(rLayerCount));

  vv_host_tensor_t& filters = *dbm.getWeights();
  vv_host_tensor_t& c = *dbm.getHiddenBiases();
  tensor_t b = *dbm.getVisibleBias(), binc = zeros<value_t>(b.size());

  v_ctensor_t cF[cLayerCount], cFinc[cLayerCount], cc[cLayerCount], ccinc[cLayerCount];
  for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
    cF[iLayer].resize(filters[iLayer]->size());
    cFinc[iLayer].resize(filters[iLayer]->size());
    cc[iLayer].resize(c[iLayer]->size());
    ccinc[iLayer].resize(c[iLayer]->size());

    outFilters->at(iLayer) = boost::make_shared<v_host_tensor_t>(filters[iLayer]->size());
    outC->at(iLayer) = boost::make_shared<v_host_tensor_t>(c[iLayer]->size());
    outMasks->at(iLayer) = boost::make_shared<host_tensor_t>(*dbm.getMasks()->at(iLayer));
  }

  matrix_t W[rLayerCount], Winc[rLayerCount], c_flat[rLayerCount], cinc_flat[rLayerCount], prods[rLayerCount], hidact[rLayerCount];
  for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
    W[iLayer] = *dbm.getWeightMatrices()->at(iLayer);
    Winc[iLayer] = zeros<value_t>(W[iLayer].size());
    c_flat[iLayer] = *dbm.getFlatBiases()->at(iLayer);
    cinc_flat[iLayer] = zeros<value_t>(c_flat[iLayer].size());
  }

  random_tensor<value_t, 2, true, normal<value_t> > flat_noise[rLayerCount];
  for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer)
    flat_noise[iLayer].resize(c_flat[iLayer].size());

  outDbm->setWeights(outFilters);
  outDbm->setHiddenBiases(outC);
  outDbm->setMasks(outMasks);
  outDbm->setVisibleBlockSize(dbm.getVisibleBlockSize());
  outDbm->setMean(dbm.getMean());
  outDbm->setStddev(dbm.getStddev());
  outDbm->setWeightMatrices(outMatrices);
  outDbm->setFlatBiases(outFlatBiases);

  // These variables will be used in both, the positive and negative phase
  tensor_t v_master[cLayerCount + 1], V_master[cLayerCount], v_diff[cLayerCount + 1];
  ctensor_t cV_master[cLayerCount];
  matrix_t v_flat[rLayerCount + 1], h_flat[rLayerCount], flat_diff[rLayerCount];

  // Monitoring effort
  value_t pos_avg[cLayerCount + rLayerCount + 1], neg_avg[cLayerCount + rLayerCount + 1];

  // Used to save the entire state of the particles used to estimate the data-independent statistics
  host_tensor_t v_particles[getSampleCount()][cLayerCount + 1];
  host_matrix_t flat_particles[getSampleCount()][rLayerCount + 1];

  #pragma omp parallel
  {
    /*** PREPARE GPU THREADS ***/

    int tid = omp_get_thread_num();
    cudaSetDevice(tid);

    // Enable peer to peer access of each card with the master card and vice versa
    if (tid == 0) {
      for (int i = 1; i < gpuCount; ++i)
        cudaDeviceEnablePeerAccess(i, 0);
    } else {
      cudaDeviceEnablePeerAccess(0, 0);
    }
    #pragma omp barrier

    plan_t plan_v[cLayerCount], iplan_v[cLayerCount], plan_h[cLayerCount], iplan_h[cLayerCount];
    tensor_t hMask[cLayerCount];

    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise[cLayerCount];
    for (size_t i = 0; i < cLayerCount; ++i)
      h_noise[i].resize(layerSize[i], tid);
    random_tensor<value_t, dimCount, true, normal<value_t> > V_noise(visSize[0], tid);

    for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
      hMask[iLayer] = *dbm.getMasks()->at(iLayer);

      // Copy filters to the device and pre-calculate the FFT
      {
        tensor_t f, h, kern, pad;
        ctensor_t cf, ch;
        for (size_t k = tid; k < filters[iLayer]->size(); k += gpuCount) {

          kern = *filters[iLayer]->at(k);
          dim_t topleft = visSize[iLayer] / 2 - kern.size() / 2;
          pad = zeros<value_t>(visSize[iLayer]);
          pad[topleft, kern.size()] = kern;
          f = ifftshift(pad, dimCount - 1);
          cf = fft(f, dimCount - 1, plan_v[iLayer]);
          cF[iLayer][k] = boost::make_shared<ctensor_t>(cf);
          cFinc[iLayer][k] = boost::make_shared<ctensor_t>(zeros<complex_t>(cf.size(), cf.fullsize()));

          h = *c[iLayer]->at(k);
          ch = fft(h, dimCount - 1, plan_h[iLayer]);
          cc[iLayer][k] = boost::make_shared<ctensor_t>(ch);
          ccinc[iLayer][k] = boost::make_shared<ctensor_t>(zeros<complex_t>(ch.size(), ch.fullsize()));
        }
      }
    }

    tensor_t h[cLayerCount], f[cLayerCount];
    ctensor_t cV[cLayerCount], ch_full[cLayerCount], ch[cLayerCount];

    #pragma omp master
    dlog(Severity::Message) << "Initialization completed. Starting training ...";

    for (size_t iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

      #pragma omp master
      {
        if (iEpoch < momentumEpochs) {
          const value_t t = (value_t)iEpoch / (value_t)momentumEpochs;
          momentum = (1 - t) * initialmomentum + t * finalmomentum;
        } else {
          momentum = finalmomentum;
        }
      }

      // Momentum is read by all threads therefore wait here until the master has done its work
      #pragma omp barrier

      for (size_t iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

        #pragma omp master
        {
          epsBatchCL = getLearningRateCL() * getLearningDecay() * batchCount / (value_t)(getLearningDecay() * batchCount + iBatch + iEpoch * batchCount);
          epsBatchRL = getLearningRateRL() * getLearningDecay() * batchCount / (value_t)(getLearningDecay() * batchCount + iBatch + iEpoch * batchCount);
          gibbsIterations = (iEpoch == 0 && iBatch == 0 ? getInitialGibbsIterations() : getGibbsIterations());
        }
        // Learning rate modifier is read by all threads
        #pragma omp barrier

        for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
          for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
            *cFinc[iLayer][k] = momentum * *cFinc[iLayer][k]; // - weightcost * *cF[k];
            *ccinc[iLayer][k] = momentum * *ccinc[iLayer][k];
          }
        }

        #pragma omp master
        {
          binc = momentum * binc;
          for (size_t iLayer = 1; iLayer < cLayerCount + 1; ++iLayer)
            v_diff[iLayer] = zeros<value_t>(hidSize[iLayer - 1]);
          v_diff[0] = zeros<value_t>(inputs[0]->size());

          for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
            Winc[iLayer] = momentum * Winc[iLayer];
            cinc_flat[iLayer] = momentum * cinc_flat[iLayer];
            flat_diff[iLayer] = zeros<value_t>(c_flat[iLayer].size());
          }

          for (size_t iLayer = 0; iLayer < cLayerCount + rLayerCount + 1; ++iLayer)
            pos_avg[iLayer] = neg_avg[iLayer] = 0;
        }

        /*** POSITIVE PHASE ***/

        for (size_t iSample = iBatch * batchSize; iSample < (iBatch + 1) * batchSize && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {

          /*** Load visible layer ***/

          cudaStreamSynchronize(0);
          #pragma omp barrier

          #pragma omp master
          {
            v_master[0] = *inputs[iSample];
            v_master[0] = (v_master[0] - dbm.getMean()) / dbm.getStddev();
            V_master[0] = rearrange(v_master[0], rearrangeBlock[0]);
            V_master[0] = V_master[0] * repeat(hMask[0], visSize[0] / layerSize[0]);
            cV_master[0] = fft(V_master[0], dimCount - 1, plan_v[0]);

            binc = binc + epsBatchCL * epsilonvb * V_master[0];

            for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer)
              v_master[iLayer + 1] = zeros<value_t>(hidSize[iLayer]);

            v_flat[0] = zeros<value_t>(seq(1, (int)v_master[cLayerCount].count()));
            for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer)
              v_flat[iLayer + 1] = zeros<value_t>(c_flat[iLayer].size());

            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          cV[0] = cV_master[0];
          cudaStreamSynchronize(0);
          #pragma omp barrier

          // Perform multiple mean field updates (first update initialize the model)
          for (size_t iMeanField = 0; iMeanField < getMeanFieldIterations(); ++iMeanField) {

            // Go through convolutional layers first
            for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {

              // If not the top-most layer, calculate top-down signal from convolutional layer
              if (iLayer < cLayerCount - 1) {

                cV[iLayer + 1] = zeros<complex_t>(cF[iLayer + 1][0]->size(), cF[iLayer + 1][0]->fullsize());

                #pragma omp master
                {
                  cV_master[iLayer + 1] = zeros<complex_t>(cF[iLayer + 1][0]->size(), cF[iLayer + 1][0]->fullsize());
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier

                for (size_t k = tid; k < cF[iLayer + 1].size(); k += gpuCount) {
                  h[iLayer + 1] = v_master[iLayer + 2][seq(0,0,0,(int)k), layerSize[iLayer + 1]];
                  ch[iLayer + 1] = fft(h[iLayer + 1], dimCount - 1, plan_h[iLayer + 1]);

                  cV[iLayer + 1] = cV[iLayer + 1] + *cF[iLayer + 1][k] * repeat(ch[iLayer + 1], cF[iLayer + 1][k]->size() / ch[iLayer + 1].size());
                }

                #pragma omp critical
                {
                  cV_master[iLayer + 1] = cV_master[iLayer + 1] + cV[iLayer + 1];
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier

                #pragma omp master
                {
                  V_master[iLayer + 1] = ifft(cV_master[iLayer + 1], dimCount - 1, iplan_v[iLayer + 1]);
                  v_master[iLayer + 1] = rearrange_r(V_master[iLayer + 1], rearrangeBlock[iLayer + 1]);
                  cudaStreamSynchronize(0);
                }
              } else {  // calculate top-down signal from first RBM
                #pragma omp master
                {
                  v_flat[0] = prod(v_flat[1], tbblas::trans(W[0]));
                  assert(v_flat[0].count() == v_master[iLayer + 1].count());
                  thrust::copy(v_flat[0].begin(), v_flat[0].end(), v_master[iLayer + 1].begin());
                  cudaStreamSynchronize(0);
                }
              }
              #pragma omp barrier

              // bottom-up signal
              for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
                if (iMeanField == 0)  // double weights because I'm getting zero input from the upper layer
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer] * 2.0;
                else
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer];
                ch[iLayer] = sum(ch_full[iLayer], dimCount - 1);
                ch[iLayer] = ch[iLayer] + *cc[iLayer][k];
                h[iLayer] = ifft(ch[iLayer], dimCount - 1, iplan_h[iLayer]);
                h[iLayer] = h[iLayer] + v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]];
                h[iLayer] = nrelu_mean(h[iLayer]);

                if (iMeanField == getMeanFieldIterations() - 1) {
                  // dF_k = ~h * v
                  ch[iLayer] = fft(h[iLayer], dimCount - 1, plan_h[iLayer]);
                  *cFinc[iLayer][k] = *cFinc[iLayer][k] + epsBatchCL * epsilonw[iLayer] * repeat(conj(ch[iLayer]), cV[iLayer].size() / ch[iLayer].size()) * cV[iLayer];
                  *ccinc[iLayer][k] = *ccinc[iLayer][k] + epsBatchCL * epsilonhb[iLayer] * ch[iLayer];
                }

                v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]] = h[iLayer] * hMask[iLayer];
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier

              if (iLayer < cLayerCount - 1) {
                // rearrange into master first and then let all threads read from master into cV
                #pragma omp master
                {
                  V_master[iLayer + 1] = rearrange(v_master[iLayer + 1], rearrangeBlock[iLayer + 1]);
                  cV_master[iLayer + 1] = fft(V_master[iLayer + 1], dimCount - 1, plan_v[iLayer + 1]);
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier
                cV[iLayer + 1] = cV_master[iLayer + 1];
              } else {
                #pragma omp master
                thrust::copy(v_master[iLayer + 1].begin(), v_master[iLayer + 1].end(), v_flat[0].begin());
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier
            }

            // Then go through RBM layer
            #pragma omp master
            {
              for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {

                // bottom-up signal
                h_flat[iLayer] = prod(v_flat[iLayer], W[iLayer]);

                if (iLayer < rLayerCount - 1) {  // add top-down signal and bias
                  v_flat[iLayer + 1] = prod(v_flat[iLayer + 2], tbblas::trans(W[iLayer + 1]));
                  if (iMeanField == 0)
                    v_flat[iLayer + 1] = nrelu_mean(v_flat[iLayer + 1] + 2.0 * h_flat[iLayer] + c_flat[iLayer]);
                  else
                    v_flat[iLayer + 1] = nrelu_mean(v_flat[iLayer + 1] + h_flat[iLayer] + c_flat[iLayer]);
                } else {                         // add bias only
                  v_flat[iLayer + 1] = nrelu_mean(h_flat[iLayer] + c_flat[iLayer]);
                }

                if (iMeanField == getMeanFieldIterations() - 1) {

                  // Update weights and biases
                  // (x_n)(mu_n)'
                  prods[iLayer] = prod(trans(v_flat[iLayer]), v_flat[iLayer + 1]);
                  Winc[iLayer] = Winc[iLayer] + epsBatchRL * prods[iLayer] / batchSize;

                  // Calculate the total activation of the hidden and visible units
                  hidact[iLayer] = sum(v_flat[iLayer + 1], 0);
                  cinc_flat[iLayer] = cinc_flat[iLayer] + epsBatchRL * hidact[iLayer] / batchSize;
                }
              }
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier
          }

          #pragma omp master
          {
            for (size_t iLayer = 0; iLayer < cLayerCount + 1; ++iLayer) {
              v_diff[iLayer] = v_diff[iLayer] + v_master[iLayer];
              pos_avg[iLayer] += sum(v_master[iLayer]) / v_master[iLayer].count() / batchSize;
            }

            for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
              flat_diff[iLayer] = flat_diff[iLayer] + v_flat[iLayer + 1];
              pos_avg[iLayer + cLayerCount + 1] += sum(v_flat[iLayer + 1]) / v_flat[iLayer + 1].count() / batchSize;
            }
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier
        }

          /*** END OF POSITIVE PHASE ***/

          /*** NEGATIVE PHASE ***/

        for (size_t iSample = 0; iSample < getSampleCount() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {

          cudaStreamSynchronize(0);
          #pragma omp barrier

          if (iEpoch == 0 && iBatch == 0) {

            #pragma omp master
            {
              V_master[0] = V_noise * repeat(hMask[0], visSize[0] / layerSize[0]);
              cV_master[0] = fft(V_master[0], dimCount - 1, plan_v[0]);

              for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer)
                v_master[iLayer + 1] = zeros<value_t>(hidSize[iLayer]);

              v_flat[0] = zeros<value_t>(seq(1, (int)v_master[cLayerCount].count()));
              for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer)
                v_flat[iLayer + 1] = zeros<value_t>(c_flat[iLayer].size());

              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            cV[0] = cV_master[0];

          } else {
            #pragma omp master
            {
              for (size_t iLayer = 0; iLayer < cLayerCount + 1; ++iLayer) {
                v_master[iLayer] = v_particles[iSample][iLayer];
                if (iLayer < cLayerCount) {
                  V_master[iLayer] = rearrange(v_master[iLayer], rearrangeBlock[iLayer]);
                  cV_master[iLayer] = fft(V_master[iLayer], dimCount - 1, plan_v[iLayer]);
                } else {
                  thrust::copy(v_master[iLayer].begin(), v_master[iLayer].end(), v_flat[0].begin());
                }
              }

              for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer)
                v_flat[iLayer + 1] = flat_particles[iSample][iLayer];
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer)
              cV[iLayer] = cV_master[iLayer];
          }
          cudaStreamSynchronize(0);
          #pragma omp barrier

          // Perform multiple Gibbs updates (first update initialize the model)
          for (size_t iGibbs = 0; iGibbs < gibbsIterations; ++iGibbs) {

            /*** Follow bottom-up Gibbs chain ***/

            for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {

              // If not the top-most layer, calculate top-down signal from convolutional layers
              if (iLayer < cLayerCount - 1) {

                cV[iLayer + 1] = zeros<complex_t>(cF[iLayer + 1][0]->size(), cF[iLayer + 1][0]->fullsize());

                #pragma omp master
                {
                  cV_master[iLayer + 1] = zeros<complex_t>(cF[iLayer + 1][0]->size(), cF[iLayer + 1][0]->fullsize());
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier

                for (size_t k = tid; k < cF[iLayer + 1].size(); k += gpuCount) {
                  h[iLayer + 1] = v_master[iLayer + 2][seq(0,0,0,(int)k), layerSize[iLayer + 1]];
                  ch[iLayer + 1] = fft(h[iLayer + 1], dimCount - 1, plan_h[iLayer + 1]);

                  cV[iLayer + 1] = cV[iLayer + 1] + *cF[iLayer + 1][k] * repeat(ch[iLayer + 1], cF[iLayer + 1][k]->size() / ch[iLayer + 1].size());
                }

                #pragma omp critical
                {
                  cV_master[iLayer + 1] = cV_master[iLayer + 1] + cV[iLayer + 1];
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier

                #pragma omp master
                {
                  V_master[iLayer + 1] = ifft(cV_master[iLayer + 1], dimCount - 1, iplan_v[iLayer + 1]);
                  v_master[iLayer + 1] = rearrange_r(V_master[iLayer + 1], rearrangeBlock[iLayer + 1]);
                  cudaStreamSynchronize(0);
                }
              } else { // calculate top-down signal from first RBM
                #pragma omp master
                {
                  v_flat[0] = prod(v_flat[1], tbblas::trans(W[0]));
                  assert(v_flat[0].count() == v_master[iLayer + 1].count());
                  thrust::copy(v_flat[0].begin(), v_flat[0].end(), v_master[iLayer + 1].begin());
                  cudaStreamSynchronize(0);
                }
              }
              #pragma omp barrier

              // bottom-up signal
              for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
                if (iEpoch == 0 && iBatch == 0 && iGibbs == 0)  // double weights because I'm getting zero input from the upper layer
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer] * 2.0;
                else
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer];
                ch[iLayer] = sum(ch_full[iLayer], dimCount - 1);
                ch[iLayer] = ch[iLayer] + *cc[iLayer][k];
                h[iLayer] = ifft(ch[iLayer], dimCount - 1, iplan_h[iLayer]);
                h[iLayer] = h[iLayer] + v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]];

                h[iLayer] = max(0.0, h[iLayer] + sqrt(sigm(h[iLayer])) * h_noise[iLayer]);
                v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]] = h[iLayer] * hMask[iLayer];
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier

              if (iLayer < cLayerCount - 1) {
                // rearrange into master first and then let all threads read from master into cV
                #pragma omp master
                {
                  V_master[iLayer + 1] = rearrange(v_master[iLayer + 1], rearrangeBlock[iLayer + 1]);
                  cV_master[iLayer + 1] = fft(V_master[iLayer + 1], dimCount - 1, plan_v[iLayer + 1]);
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier
                cV[iLayer + 1] = cV_master[iLayer + 1];
              } else {
                #pragma omp master
                thrust::copy(v_master[iLayer + 1].begin(), v_master[iLayer + 1].end(), v_flat[0].begin());
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier
            } /* end of bottom-up pass */

            // Then go through RBM layer
            #pragma omp master
            {
              for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {

                // bottom-up signal
                h_flat[iLayer] = prod(v_flat[iLayer], W[iLayer]);

                if (iLayer < rLayerCount - 1) {  // add top-down signal and bias
                  v_flat[iLayer + 1] = prod(v_flat[iLayer + 2], tbblas::trans(W[iLayer + 1]));
                  if (iGibbs == 0 && iEpoch == 0 && iBatch == 0)
                    v_flat[iLayer + 1] = v_flat[iLayer + 1] + 2.0 * h_flat[iLayer] + c_flat[iLayer];
                  else
                    v_flat[iLayer + 1] = v_flat[iLayer + 1] + h_flat[iLayer] + c_flat[iLayer];
                } else {                         // add bias only
                  v_flat[iLayer + 1] = h_flat[iLayer] + c_flat[iLayer];
                }
                v_flat[iLayer + 1] = max(0.0, v_flat[iLayer + 1] + sqrt(sigm(v_flat[iLayer + 1])) * flat_noise[iLayer]);

                if (iLayer == rLayerCount - 1 && iGibbs == gibbsIterations - 1) {

                  // (x_n)(mu_n)'
                  prods[iLayer] = prod(trans(v_flat[iLayer]), v_flat[iLayer + 1]);
                  Winc[iLayer] = Winc[iLayer] - epsBatchRL * prods[iLayer] / batchSize;

                  // Calculate the total activation of the hidden and visible units
                  hidact[iLayer] = sum(v_flat[iLayer + 1], 0);
                  cinc_flat[iLayer] = cinc_flat[iLayer] - epsBatchRL * hidact[iLayer] / batchSize;
                }
              }
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            /*** Follow top-down Gibbs chain ***/

            // Update RBM layers first
            #pragma omp master
            {
              for (int iLayer = rLayerCount - 2; iLayer >= 0; --iLayer) {

                // bottom-up signal
                h_flat[iLayer] = prod(v_flat[iLayer], W[iLayer]);

                v_flat[iLayer + 1] = prod(v_flat[iLayer + 2], tbblas::trans(W[iLayer + 1]));
                v_flat[iLayer + 1] = v_flat[iLayer + 1] + h_flat[iLayer] + c_flat[iLayer];
                v_flat[iLayer + 1] = max(0.0, v_flat[iLayer + 1] + sqrt(sigm(v_flat[iLayer + 1])) * flat_noise[iLayer]);

                if (iGibbs == gibbsIterations - 1 && iLayer + 1 < rLayerCount - 1) {

                  // (x_n)(mu_n)'
                  prods[iLayer + 1] = prod(trans(v_flat[iLayer + 1]), v_flat[iLayer + 2]);
                  Winc[iLayer + 1] = Winc[iLayer + 1] - epsBatchRL * prods[iLayer + 1] / batchSize;

                  // Calculate the total activation of the hidden and visible units
                  hidact[iLayer + 1] = sum(v_flat[iLayer + 2], 0);
                  cinc_flat[iLayer + 1] = cinc_flat[iLayer + 1] - epsBatchRL * hidact[iLayer + 1] / batchSize;
                }
              }
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            for (int iLayer = cLayerCount; iLayer >= 0; --iLayer) {

              // if not top-most layer, calculate top-down signal from convolutional layers
              if (iLayer < cLayerCount) {
                cV[iLayer] = zeros<complex_t>(cF[iLayer][0]->size(), cF[iLayer][0]->fullsize());

                #pragma omp master
                {
                  cV_master[iLayer] = zeros<complex_t>(cF[iLayer][0]->size(), cF[iLayer][0]->fullsize());
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier

                for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
                  h[iLayer] = v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]];
                  ch[iLayer] = fft(h[iLayer], dimCount - 1, plan_h[iLayer]);
                  cV[iLayer] = cV[iLayer] + *cF[iLayer][k] * repeat(ch[iLayer], cF[iLayer][k]->size() / ch[iLayer].size());
                }

                #pragma omp critical
                {
                  cV_master[iLayer] = cV_master[iLayer] + cV[iLayer];
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier

                #pragma omp master
                {
                  V_master[iLayer] = ifft(cV_master[iLayer], dimCount - 1, iplan_v[iLayer]);
                  if (iLayer == 0) {
                    V_master[0] = (V_master[0] + b) * repeat(hMask[0], visSize[0] / layerSize[0]);

                    if (iGibbs == gibbsIterations - 1) {
                      binc = binc -  epsBatchCL * epsilonvb * V_master[0];
                    }
                  }

                  v_master[iLayer] = rearrange_r(V_master[iLayer], rearrangeBlock[iLayer]);
                  cudaStreamSynchronize(0);
                }
              } else { // calculate top-down signal from first RBM
                #pragma omp master
                {
                  v_flat[0] = prod(v_flat[1], tbblas::trans(W[0]));
                  assert(v_flat[0].count() == v_master[iLayer].count());
                  thrust::copy(v_flat[0].begin(), v_flat[0].end(), v_master[iLayer].begin());
                  cudaStreamSynchronize(0);
                }
              }
              #pragma omp barrier

              // bottom-up signal
              if (iLayer > 0) {
                for (size_t k = tid; k < cF[iLayer - 1].size(); k += gpuCount) {
                  ch_full[iLayer - 1] = conj(*cF[iLayer - 1][k]) * cV[iLayer - 1];
                  ch[iLayer - 1] = sum(ch_full[iLayer - 1], dimCount - 1);
                  ch[iLayer - 1] = ch[iLayer - 1] + *cc[iLayer - 1][k];
                  h[iLayer - 1] = ifft(ch[iLayer - 1], dimCount - 1, iplan_h[iLayer - 1]);
                  h[iLayer - 1] = h[iLayer - 1] + v_master[iLayer][seq(0,0,0,(int)k), layerSize[iLayer - 1]];
                  h[iLayer - 1] = max(0.0, h[iLayer - 1] + sqrt(sigm(h[iLayer - 1])) * h_noise[iLayer - 1]);

                  if (iGibbs == gibbsIterations - 1) {
                    // dF_k = ~h * v
                    ch[iLayer - 1] = fft(h[iLayer - 1], dimCount - 1, plan_h[iLayer - 1]);
                    *cFinc[iLayer - 1][k] = *cFinc[iLayer - 1][k] - epsBatchCL * epsilonw[iLayer - 1] * repeat(conj(ch[iLayer - 1]), cV[iLayer - 1].size() / ch[iLayer - 1].size()) * cV[iLayer - 1];
                    *ccinc[iLayer - 1][k] = *ccinc[iLayer - 1][k] - epsBatchCL * epsilonhb[iLayer - 1] * ch[iLayer - 1];
                  }

                  v_master[iLayer][seq(0,0,0,(int)k), layerSize[iLayer - 1]] = h[iLayer - 1] * hMask[iLayer - 1];
                }
                cudaStreamSynchronize(0);
                #pragma omp barrier
              }

              if (iLayer < cLayerCount) {
                // rearrange into master first and then let all threads read from master into cV
                #pragma omp master
                {
                  V_master[iLayer] = rearrange(v_master[iLayer], rearrangeBlock[iLayer]);
                  cV_master[iLayer] = fft(V_master[iLayer], dimCount - 1, plan_v[iLayer]);
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier
                cV[iLayer] = cV_master[iLayer];
              } else {
                #pragma omp master
                thrust::copy(v_master[iLayer].begin(), v_master[iLayer].end(), v_flat[0].begin());

                if (iGibbs == gibbsIterations - 1) {
                  // Update weights and bias terms of W[0]
                  // (x_n)(mu_n)'
                  prods[0] = prod(trans(v_flat[0]), v_flat[1]);
                  Winc[0] = Winc[0] - epsBatchRL * prods[0] / batchSize;

                  // Calculate the total activation of the hidden and visible units
                  hidact[0] = sum(v_flat[1], 0);
                  cinc_flat[0] = cinc_flat[0] - epsBatchRL * hidact[0] / batchSize;
                }
              }

              cudaStreamSynchronize(0);
              #pragma omp barrier
            }
          }

          // Save state of the sample
          #pragma omp master
          {
            for (size_t iLayer = 0; iLayer < cLayerCount + 1; ++iLayer) {
              if (iLayer < cLayerCount)
                v_master[iLayer] = rearrange_r(V_master[iLayer], rearrangeBlock[iLayer]);
              v_particles[iSample][iLayer] = v_master[iLayer];
              v_diff[iLayer] = v_diff[iLayer] - v_master[iLayer];
              neg_avg[iLayer] += sum(v_master[iLayer]) / v_master[iLayer].count() / batchSize;
            }
            for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
              flat_particles[iSample][iLayer] = v_flat[iLayer + 1];
              flat_diff[iLayer] = flat_diff[iLayer] - v_flat[iLayer + 1];
              neg_avg[iLayer + cLayerCount + 1] += sum(v_flat[iLayer + 1]) / v_flat[iLayer + 1].count() / batchSize;
            }
            cudaStreamSynchronize(0);
            
            if (iEpoch == 0 && iBatch == 0)
              dlog(Severity::Trace) << "Gibbs chain " << iSample + 1 << " initialized.";
          }
          #pragma omp barrier

        } /* end of samples */

          /*** END OF NEGATIVE PHASE ***/

        // Apply gradient
        for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
          complex_t avg_f = 0, avg_c = 0;
          for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {

            avg_f = avg_f + sum(*cFinc[iLayer][k]);
            avg_c = avg_c + sum(*ccinc[iLayer][k]);

            f[iLayer] = ifft(*cFinc[iLayer][k], dimCount - 1, iplan_v[iLayer]);
            f[iLayer] = f[iLayer] * mask<value_t>(f[iLayer].size(), filters[iLayer]->at(0)->size());
            *cFinc[iLayer][k] = fft(f[iLayer], dimCount - 1, plan_v[iLayer]);
            *cF[iLayer][k] = *cF[iLayer][k] + *cFinc[iLayer][k];
            *cc[iLayer][k] = *cc[iLayer][k] + *ccinc[iLayer][k];
          }
          #pragma omp master
          std::cout << "Filters:";
          #pragma omp barrier

          #pragma omp critical
          std::cout << " " << avg_f;
          #pragma omp barrier

          #pragma omp master
          std::cout << std::endl;
          #pragma omp barrier

          #pragma omp master
          std::cout << "Biases:";
          #pragma omp barrier

          #pragma omp critical
          std::cout << " " << avg_c;
          #pragma omp barrier

          #pragma omp master
          std::cout << std::endl;
          #pragma omp barrier
        }
        #pragma omp master
        {
          b = b + binc;
          std::cout << "VB: " << sum(binc) << std::endl;

          for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
            std::cout << "W: " << sum(Winc[iLayer]) << std::endl;
            std::cout << "c: " << sum(cinc_flat[iLayer]) << std::endl;
            W[iLayer] = W[iLayer] + Winc[iLayer];
            c_flat[iLayer] = c_flat[iLayer] + cinc_flat[iLayer];
          }
        }

        cudaStreamSynchronize(0);
        #pragma omp barrier

        #pragma omp master
        {
          std::stringstream errors;

          for (size_t iLayer = 0; iLayer < cLayerCount + 1; ++iLayer)
            errors << sum(abs(v_diff[iLayer])) / v_diff[iLayer].count() / batchSize << " ";

          for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer)
            errors << sum(abs(flat_diff[iLayer])) / flat_diff[iLayer].count() / batchSize << " ";

          dlog(Severity::Message) << "Error at epoch " << iEpoch + 1 << " batch " << iBatch + 1 << ": " << errors.str();

          for (size_t iLayer = 0; iLayer < cLayerCount + rLayerCount + 1; ++iLayer)
            std::cout << "Layer " << iLayer + 1 << ": " << pos_avg[iLayer] << " : " << neg_avg[iLayer] << std::endl;

          if (monitor)
            monitor->reportProgress(100. * (iEpoch * batchCount + iBatch + 1) / (getEpochCount() * batchCount));
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier
      }
    } /* end of epochs */

    // Free up memory
    for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
      for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
        cFinc[iLayer][k] = ccinc[iLayer][k] = boost::shared_ptr<ctensor_t>();
      }
    }

    // Save model
    for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
      tensor_t hb, p, k;
      for (size_t i = tid; i < cF[iLayer].size(); i += gpuCount) {
        dim_t topleft = visSize[iLayer] / 2 - filters[iLayer]->at(0)->size() / 2;

        f[iLayer] = ifft(*cF[iLayer][i], dimCount - 1, iplan_v[iLayer]);
        p = fftshift(f[iLayer], dimCount - 1);
        k = p[topleft, filters[iLayer]->at(0)->size()];
        outFilters->at(iLayer)->at(i) = boost::make_shared<host_tensor_t>(k);

        hb = ifft(*cc[iLayer][i], dimCount - 1, iplan_h[iLayer]);
        hb = hb * (abs(hb) > 1e-16);
        outC->at(iLayer)->at(i) = boost::make_shared<host_tensor_t>(hb);

        cF[iLayer][i] = cc[iLayer][i] = boost::shared_ptr<ctensor_t>();
      }
    }

    #pragma omp master
    {
      outDbm->setVisibleBias(boost::make_shared<host_tensor_t>(b));

      for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
        outMatrices->at(iLayer) = boost::make_shared<host_matrix_t>(W[iLayer]);
        outFlatBiases->at(iLayer) = boost::make_shared<host_matrix_t>(c_flat[iLayer]);
      }
    }

    if (tid == 0) {
      for (int i = 1; i < gpuCount; ++i)
        cudaDeviceDisablePeerAccess(i);
    } else {
      cudaDeviceDisablePeerAccess(0);
    }

  } /* end of parallel */

  newState->setOutputModel(outDbm);
}

}

}


