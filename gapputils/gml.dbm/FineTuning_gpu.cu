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
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(MeanFieldIterations, test);
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

  // Get inputs
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getDataset();

  // Load model into device memory
  Model& dbm = *getInitialModel();

  // A DBM with 1 visible layer and n hidden layers has n layers for the sake of writing this code
  size_t layerCount = dbm.getWeights()->size();
  assert(layerCount);

  dim_t visSize[layerCount], hidSize[layerCount], layerSize[layerCount];
  for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
    visSize[iLayer] = hidSize[iLayer] = layerSize[iLayer] = dbm.getHiddenBiases()->at(iLayer)->at(0)->size();
    visSize[iLayer][dimCount - 1] = dbm.getWeights()->at(iLayer)->at(0)->size()[dimCount - 1];
    hidSize[iLayer][dimCount - 1] = dbm.getWeights()->at(iLayer)->size();
  }

  dim_t rearrangeBlock[layerCount];
  rearrangeBlock[0] = inputs[0]->size() / visSize[0];
  rearrangeBlock[0][dimCount - 1] = 1;
  assert(rearrangeBlock[0] == dbm.getVisibleBlockSize());
  for (size_t iLayer = 1; iLayer < layerCount; ++iLayer)
    rearrangeBlock[iLayer] = dbm.getHiddenBiases()->at(iLayer - 1)->at(0)->size() / dbm.getHiddenBiases()->at(iLayer)->at(0)->size();

  const size_t batchSize = getBatchSize();
  const size_t batchCount = inputs.size() / batchSize;

  // Initialize constants
  value_t epsilonw[layerCount], epsilonhb[layerCount], epsilonvb;
  value_t epsBatch;
  for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
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
  boost::shared_ptr<vv_host_tensor_t> outFilters(new vv_host_tensor_t(layerCount));
  boost::shared_ptr<vv_host_tensor_t> outC(new vv_host_tensor_t(layerCount));
  boost::shared_ptr<v_host_tensor_t> outMasks(new v_host_tensor_t(layerCount));

  vv_host_tensor_t& filters = *dbm.getWeights();
  vv_host_tensor_t& c = *dbm.getHiddenBiases();
  tensor_t b = *dbm.getVisibleBias(), binc = zeros<value_t>(b.size());

  v_ctensor_t cF[layerCount], cFinc[layerCount], cc[layerCount], ccinc[layerCount];
  for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
    cF[iLayer].resize(filters[iLayer]->size());
    cFinc[iLayer].resize(filters[iLayer]->size());
    cc[iLayer].resize(c[iLayer]->size());
    ccinc[iLayer].resize(c[iLayer]->size());

    outFilters->at(iLayer) = boost::make_shared<v_host_tensor_t>(filters[iLayer]->size());
    outC->at(iLayer) = boost::make_shared<v_host_tensor_t>(c[iLayer]->size());
    outMasks->at(iLayer) = boost::make_shared<host_tensor_t>(*dbm.getMasks()->at(iLayer));
  }

  outDbm->setWeights(outFilters);
  outDbm->setHiddenBiases(outC);
  outDbm->setMasks(outMasks);
  outDbm->setVisibleBlockSize(dbm.getVisibleBlockSize());
  outDbm->setMean(dbm.getMean());
  outDbm->setStddev(dbm.getStddev());

  // These variables will be used in both, the positive and negative phase
  tensor_t v_master[layerCount + 1], V_master[layerCount], v_diff[layerCount + 1];
  ctensor_t cV_master[layerCount];

  // Used to save the entire state of the particles used to estimate the data-independent statistics
  host_tensor_t v_particles[getSampleCount()][layerCount + 1];

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

    plan_t plan_v[layerCount], iplan_v[layerCount], plan_h[layerCount], iplan_h[layerCount];
    tensor_t hMask[layerCount];

    random_tensor<value_t, dimCount, true, normal<value_t> > h_noise[layerCount];
    for (size_t i = 0; i < layerCount; ++i)
      h_noise[i].resize(layerSize[i], tid);
    random_tensor<value_t, dimCount, true, normal<value_t> > V_noise(visSize[0], tid);

    for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
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

    tensor_t h[layerCount], f[layerCount];
    ctensor_t cV[layerCount], ch_full[layerCount], ch[layerCount];

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
        epsBatch = getLearningRate() * getLearningDecay() * batchCount / (value_t)(getLearningDecay() * batchCount + iBatch + iEpoch * batchCount);
        // Learning rate modifier is read by all threads
        #pragma omp barrier

        for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
          for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
            *cFinc[iLayer][k] = momentum * *cFinc[iLayer][k]; // - weightcost * *cF[k];
            *ccinc[iLayer][k] = momentum * *ccinc[iLayer][k];
          }
        }

        #pragma omp master
        {
          binc = momentum * binc;
          for (size_t iLayer = 1; iLayer < layerCount + 1; ++iLayer)
            v_diff[iLayer] = zeros<value_t>(hidSize[iLayer - 1]);
          v_diff[0] = zeros<value_t>(inputs[0]->size());
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

            binc = binc + epsBatch * epsilonvb * V_master[0];

            for (size_t iLayer = 0; iLayer < layerCount; ++iLayer)
              v_master[iLayer + 1] = zeros<value_t>(hidSize[iLayer]);

            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          cV[0] = cV_master[0];

          // Perform multiple mean field updates (first update initialize the model)
          for (size_t iMeanField = 0; iMeanField < getMeanFieldIterations(); ++iMeanField) {
            for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {

              // If not the top-most layer, calculate top-down signal
              if (iLayer < layerCount - 1) {

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
                #pragma omp barrier
              }

              // bottom-up signal
              for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
                if (iMeanField == 0 && iLayer < layerCount - 1)  // double weights because I'm getting zero input from the upper layer
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer] * 2.0;
                else
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer];
                ch[iLayer] = sum(ch_full[iLayer], dimCount - 1);
                ch[iLayer] = ch[iLayer] + *cc[iLayer][k];
                h[iLayer] = ifft(ch[iLayer], dimCount - 1, iplan_h[iLayer]);
                if (iLayer < layerCount - 1)
                  h[iLayer] = h[iLayer] + v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]];
                h[iLayer] = nrelu_mean(h[iLayer]);

                if (iMeanField == getMeanFieldIterations() - 1) {
                  // dF_k = ~h * v
                  ch[iLayer] = fft(h[iLayer], dimCount - 1, plan_h[iLayer]);
                  *cFinc[iLayer][k] = *cFinc[iLayer][k] + epsBatch * epsilonw[iLayer] * repeat(conj(ch[iLayer]), cV[iLayer].size() / ch[iLayer].size()) * cV[iLayer];
                  *ccinc[iLayer][k] = *ccinc[iLayer][k] + epsBatch * epsilonhb[iLayer] * ch[iLayer];
                }

                v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]] = h[iLayer] * hMask[iLayer];
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier

              if (iLayer < layerCount - 1) {
                // rearrange into master first and then let all threads read from master into cV
                #pragma omp master
                {
                  V_master[iLayer + 1] = rearrange(v_master[iLayer + 1], rearrangeBlock[iLayer + 1]);
                  cV_master[iLayer + 1] = fft(V_master[iLayer + 1], dimCount - 1, plan_v[iLayer + 1]);
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier
                cV[iLayer + 1] = cV_master[iLayer + 1];
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier
            }
          }

          #pragma omp master
          {
            for (size_t iLayer = 0; iLayer < layerCount + 1; ++iLayer)
              v_diff[iLayer] = v_diff[iLayer] + v_master[iLayer];
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

              for (size_t iLayer = 0; iLayer < layerCount; ++iLayer)
                v_master[iLayer + 1] = zeros<value_t>(hidSize[iLayer]);
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            cV[0] = cV_master[0];

          } else {
            #pragma omp master
            {
              for (size_t iLayer = 0; iLayer < layerCount + 1; ++iLayer) {
                v_master[iLayer] = v_particles[iSample][iLayer];
                if (iLayer < layerCount) {
                  V_master[iLayer] = rearrange(v_master[iLayer], rearrangeBlock[iLayer]);
                  cV_master[iLayer] = fft(V_master[iLayer], dimCount - 1, plan_v[iLayer]);
                }
              }
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier

            for (size_t iLayer = 0; iLayer < layerCount; ++iLayer)
              cV[iLayer] = cV_master[iLayer];
          }

          // Perform multiple Gibbs updates (first update initialize the model)
          for (size_t iGibbs = 0; iGibbs < getGibbsIterations(); ++iGibbs) {

            /*** Follow bottom-up Gibbs chain ***/

            for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {

              // If not the top-most layer, calculate top-down signal
              if (iLayer < layerCount - 1) {

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
                #pragma omp barrier
              }

              // bottom-up signal
              for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
                if (iEpoch == 0 && iBatch == 0 && iGibbs == 0 && iLayer < layerCount - 1)  // double weights because I'm getting zero input from the upper layer
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer] * 2.0;
                else
                  ch_full[iLayer] = conj(*cF[iLayer][k]) * cV[iLayer];
                ch[iLayer] = sum(ch_full[iLayer], dimCount - 1);
                ch[iLayer] = ch[iLayer] + *cc[iLayer][k];
                h[iLayer] = ifft(ch[iLayer], dimCount - 1, iplan_h[iLayer]);
                if (iLayer < layerCount - 1)
                  h[iLayer] = h[iLayer] + v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]];

                h[iLayer] = max(0.0, h[iLayer] + sqrt(sigm(h[iLayer])) * h_noise[iLayer]);

                if (iGibbs == getGibbsIterations() - 1 && iLayer == layerCount - 1) {
                  // dF_k = ~h * v
                  ch[iLayer] = fft(h[iLayer], dimCount - 1, plan_h[iLayer]);
                  *cFinc[iLayer][k] = *cFinc[iLayer][k] - epsBatch * epsilonw[iLayer] * repeat(conj(ch[iLayer]), cV[iLayer].size() / ch[iLayer].size()) * cV[iLayer];
                  *ccinc[iLayer][k] = *ccinc[iLayer][k] - epsBatch * epsilonhb[iLayer] * ch[iLayer];
                }

                v_master[iLayer + 1][seq(0,0,0,(int)k), layerSize[iLayer]] = h[iLayer] * hMask[iLayer];
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier

              if (iLayer < layerCount - 1) {
                // rearrange into master first and then let all threads read from master into cV
                #pragma omp master
                {
                  V_master[iLayer + 1] = rearrange(v_master[iLayer + 1], rearrangeBlock[iLayer + 1]);
                  cV_master[iLayer + 1] = fft(V_master[iLayer + 1], dimCount - 1, plan_v[iLayer + 1]);
                  cudaStreamSynchronize(0);
                }
                #pragma omp barrier
                cV[iLayer + 1] = cV_master[iLayer + 1];
              }
              cudaStreamSynchronize(0);
              #pragma omp barrier
            } /* end of bottom-up pass */

            /*** Follow top-down Gibbs chain ***/

            for (int iLayer = layerCount - 1; iLayer >= 0; --iLayer) {

              // Calculate top-down signal
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

                  if (iGibbs == getGibbsIterations() - 1) {
                    binc = binc -  epsBatch * epsilonvb * V_master[0];
                  }
                }

                v_master[iLayer] = rearrange_r(V_master[iLayer], rearrangeBlock[iLayer]);
                cudaStreamSynchronize(0);
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

                  if (iGibbs == getGibbsIterations() - 1) {
                    // dF_k = ~h * v
                    ch[iLayer - 1] = fft(h[iLayer - 1], dimCount - 1, plan_h[iLayer - 1]);
                    *cFinc[iLayer - 1][k] = *cFinc[iLayer - 1][k] - epsBatch * epsilonw[iLayer] * repeat(conj(ch[iLayer - 1]), cV[iLayer - 1].size() / ch[iLayer - 1].size()) * cV[iLayer - 1];
                    *ccinc[iLayer - 1][k] = *ccinc[iLayer - 1][k] - epsBatch * epsilonhb[iLayer] * ch[iLayer - 1];
                  }

                  v_master[iLayer][seq(0,0,0,(int)k), layerSize[iLayer - 1]] = h[iLayer - 1] * hMask[iLayer - 1];
                }
                cudaStreamSynchronize(0);
                #pragma omp barrier
              }

              // rearrange into master first and then let all threads read from master into cV
              #pragma omp master
              {
                V_master[iLayer] = rearrange(v_master[iLayer], rearrangeBlock[iLayer]);
                cV_master[iLayer] = fft(V_master[iLayer], dimCount - 1, plan_v[iLayer]);
                cudaStreamSynchronize(0);
              }
              #pragma omp barrier
              cV[iLayer] = cV_master[iLayer];

              cudaStreamSynchronize(0);
              #pragma omp barrier
            }
          }

          // Save state of the sample
          #pragma omp master
          {
            for (size_t iLayer = 0; iLayer < layerCount + 1; ++iLayer) {
              if (iLayer < layerCount)
                v_master[iLayer] = rearrange_r(V_master[iLayer], rearrangeBlock[iLayer]);
              v_particles[iSample][iLayer] = v_master[iLayer];
              v_diff[iLayer] = v_diff[iLayer] - v_master[iLayer];
            }
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

        } /* end of samples */

          /*** END OF NEGATIVE PHASE ***/

        for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
          for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
            f[iLayer] = ifft(*cFinc[iLayer][k], dimCount - 1, iplan_v[iLayer]);
            f[iLayer] = f[iLayer] * mask<value_t>(f[iLayer].size(), filters[iLayer]->at(0)->size());
            *cFinc[iLayer][k] = fft(f[iLayer], dimCount - 1, plan_v[iLayer]);
            *cF[iLayer][k] = *cF[iLayer][k] + *cFinc[iLayer][k];
            *cc[iLayer][k] = *cc[iLayer][k] + *ccinc[iLayer][k];
          }
        }
        #pragma omp master
        b = b + binc;

        cudaStreamSynchronize(0);
        #pragma omp barrier

        #pragma omp master
        {
          std::stringstream errors;

          for (int iLayer = 0; iLayer < layerCount + 1; ++iLayer)
            errors << sum(abs(v_diff[iLayer])) / v_diff[iLayer].count() / batchSize << " ";
          dlog(Severity::Message) << "Error at epoch " << iEpoch + 1 << " batch " << iBatch + 1 << ": " << errors.str() << "(eps = " << epsBatch << ")";

          if (monitor)
            monitor->reportProgress(100. * (iEpoch * batchCount + iBatch + 1) / (getEpochCount() * batchCount));
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier
      }
    } /* end of epochs */

    // Free up memory
    for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
      for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
        cFinc[iLayer][k] = ccinc[iLayer][k] = boost::shared_ptr<ctensor_t>();
      }
    }

    for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
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
    outDbm->setVisibleBias(boost::make_shared<host_tensor_t>(b));

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


