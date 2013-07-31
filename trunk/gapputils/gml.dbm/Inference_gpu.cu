/*
 * Inference_gpu.cu
 *
 *  Created on: Jul 12, 2013
 *      Author: tombr
 */

#include "Inference.h"

#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/shift.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/linalg.hpp>

#include <omp.h>

#include "math.hpp"

namespace gml {

namespace dbm {

InferenceChecker::InferenceChecker() {
  Inference test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Mode, test);
  CHECK_MEMORY_LAYOUT2(ObservedLayer, test);
  CHECK_MEMORY_LAYOUT2(QueryLayer, test);
  CHECK_MEMORY_LAYOUT2(Iterations, test);
  CHECK_MEMORY_LAYOUT2(GpuCount, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Inference::update(IProgressMonitor* monitor) const {
  // Perform a single up pass to initialize the values
  // Iteratively update

  using namespace tbblas;

  Logbook& dlog = getLogbook();

  const unsigned dimCount = Model::dimCount;
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
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();

  // Prepare outputs
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  // Load model into device memory
  Model& dbm = *getModel();

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
  rearrangeBlock[0] = dbm.getVisibleBlockSize();
  for (size_t iLayer = 1; iLayer < cLayerCount; ++iLayer)
    rearrangeBlock[iLayer] = dbm.getHiddenBiases()->at(iLayer - 1)->at(0)->size() / dbm.getHiddenBiases()->at(iLayer)->at(0)->size();

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  const int gpuCount = getGpuCount();

  // Setting defaults

  int observedLayer, queryLayer;

  switch (getMode()) {
  case InferenceMode::BottomUp:
    observedLayer = (getObservedLayer() < 0 ? 0 : getObservedLayer());
    queryLayer = (getQueryLayer() < 0 ? cLayerCount + rLayerCount : getQueryLayer());
    break;

  case InferenceMode::TopDown:
    observedLayer = (getObservedLayer() < 0 ? cLayerCount + rLayerCount : getObservedLayer());
    queryLayer = (getQueryLayer() < 0 ? 0 : getQueryLayer());
    break;
  }

  // Perform sanity checks

  if (deviceCount < gpuCount) {
    dlog(Severity::Warning) << "Only " << deviceCount << " CUDA-enabled devices found, where " << gpuCount << " are required according to GpuCount. Aborting!";
    return;
  }

  switch (getMode()) {
  case InferenceMode::BottomUp:
    if (observedLayer >= queryLayer) {
      dlog(Severity::Warning) << "The observed layer must be below the query layer for bottom-up inference. Aborting!";
      return;
    }
    break;

  case InferenceMode::TopDown:
    if (observedLayer <= queryLayer) {
      dlog(Severity::Warning) << "The observed layer must be above the query layer for top-down inference. Aborting!";
      return;
    }
    break;
  }

  assert(omp_get_num_threads() == 1);

  cudaSetDevice(0);
  omp_set_dynamic(0);
  omp_set_num_threads(gpuCount);

  vv_host_tensor_t& filters = *dbm.getWeights();
  vv_host_tensor_t& c = *dbm.getHiddenBiases();
  tensor_t b = *dbm.getVisibleBias();

  v_ctensor_t cF[cLayerCount], cc[cLayerCount];
  for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
    cF[iLayer].resize(filters[iLayer]->size());
    cc[iLayer].resize(filters[iLayer]->size());
  }

  matrix_t W[rLayerCount], c_flat[rLayerCount];
  for (size_t iLayer = 0; iLayer < rLayerCount; ++iLayer) {
    W[iLayer] = *dbm.getWeightMatrices()->at(iLayer);
    c_flat[iLayer] = *dbm.getFlatBiases()->at(iLayer);
  }

  tensor_t v_master[cLayerCount + 1], V_master[cLayerCount];
  ctensor_t cV_master[cLayerCount];
  matrix_t v_flat[rLayerCount + 1], h_flat[rLayerCount];

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

          h = *c[iLayer]->at(k);
          ch = fft(h, dimCount - 1, plan_h[iLayer]);
          cc[iLayer][k] = boost::make_shared<ctensor_t>(ch);
        }
      }
    }

    tensor_t h[cLayerCount];
    ctensor_t cV[cLayerCount], ch_full[cLayerCount], ch[cLayerCount];

    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {

      if (getMode() == InferenceMode::BottomUp) {

        /*** LOAD VISIBLE LAYER ***/

        cudaStreamSynchronize(0);
        #pragma omp barrier

        #pragma omp master
        {
          // Initialize the flat layer first because this also reserves space in case the flat layer is observed
          for (size_t iLayer = max(0, observedLayer - (int)cLayerCount); iLayer < rLayerCount + 1; ++iLayer) {
            if (iLayer == 0)
              v_flat[iLayer] = zeros<value_t>(seq(1, (int)hidSize[cLayerCount - 1].count()));
            else
              v_flat[iLayer] = zeros<value_t>(c_flat[iLayer - 1].size());
          }

          if (observedLayer < cLayerCount) {
            v_master[observedLayer] = *inputs[i];
            V_master[observedLayer] = rearrange(v_master[observedLayer], rearrangeBlock[observedLayer]);
            if (observedLayer == 0)
              V_master[observedLayer] = (V_master[observedLayer] - dbm.getMean()) / dbm.getStddev();
            V_master[observedLayer] = V_master[observedLayer] * repeat(hMask[observedLayer], visSize[observedLayer] / layerSize[observedLayer]);
            cV_master[observedLayer] = fft(V_master[observedLayer], dimCount - 1, plan_v[observedLayer]);

            for (size_t iLayer = observedLayer; iLayer < cLayerCount; ++iLayer)
              v_master[iLayer + 1] = zeros<value_t>(hidSize[iLayer]);
          } else {
            thrust::copy(inputs[i]->begin(), inputs[i]->end(), v_flat[observedLayer - cLayerCount].begin());
          }

          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        if (observedLayer < cLayerCount)
          cV[observedLayer] = cV_master[observedLayer];

        // Perform multiple mean field updates (first update initializes the model)
        for (size_t iMeanField = 0; iMeanField < getIterations(); ++iMeanField) {

          // Go through convolutional layers first
          for (size_t iLayer = observedLayer; iLayer < cLayerCount; ++iLayer) {

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
            for (size_t iLayer = std::max(0, observedLayer - (int)cLayerCount); iLayer < rLayerCount; ++iLayer) {

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
            }
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier
        }

        #pragma omp master
        {
          if (queryLayer <= cLayerCount) {
            outputs->push_back(boost::make_shared<host_tensor_t>(v_master[queryLayer]));
          } else {
            boost::shared_ptr<host_tensor_t> output(new host_tensor_t(1, 1, 1, v_flat[queryLayer - cLayerCount].count()));
            thrust::copy(v_flat[queryLayer - cLayerCount].begin(), v_flat[queryLayer - cLayerCount].end(), output->begin());
            outputs->push_back(output);
          }
        }
      } else if (getMode() == InferenceMode::TopDown) { /*** Top-down case ***/

        /*** LOAD HIDDEN LAYER ***/

        for (size_t iLayer = 0; iLayer < min(observedLayer - 1, (int)cLayerCount); ++iLayer)
          cV[iLayer] = zeros<complex_t>(cF[iLayer][0]->size(), cF[iLayer][0]->fullsize());

        #pragma omp master
        {

          for (int iLayer = 0; iLayer < observedLayer - (int)cLayerCount + 1; ++iLayer) {
            if (iLayer == 0)
              v_flat[iLayer] = zeros<value_t>(seq(1, (int)hidSize[cLayerCount - 1].count()));
            else
              v_flat[iLayer] = zeros<value_t>(c_flat[iLayer - 1].size());
          }

          if (observedLayer > cLayerCount) {
            thrust::copy(inputs[i]->begin(), inputs[i]->end(), v_flat[observedLayer - cLayerCount].begin());
          } else {
            v_master[observedLayer] = *inputs[i];
          }

          if (observedLayer > cLayerCount)
            v_master[cLayerCount] = zeros<value_t>(hidSize[cLayerCount - 1]);

          for (size_t iLayer = 0; iLayer < min(observedLayer - 1, (int)cLayerCount); ++iLayer) {
            cV_master[iLayer] = zeros<complex_t>(cF[iLayer][0]->size(), cF[iLayer][0]->fullsize());
          }
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        // Perform multiple mean field updates (first update initialize the model)
        for (size_t iMeanField = 0; iMeanField < getIterations(); ++iMeanField) {

          // Update RBM layers first
          // this will also update v_master[cLayerCount]
          #pragma omp master
          {
            for (int iLayer = observedLayer - cLayerCount - 2; iLayer >= 0; --iLayer) {

              // bottom-up signal
              h_flat[iLayer] = prod(v_flat[iLayer], W[iLayer]);

              v_flat[iLayer + 1] = prod(v_flat[iLayer + 2], tbblas::trans(W[iLayer + 1]));
              if (iMeanField == 0)
                v_flat[iLayer + 1] = nrelu_mean(2.0 * v_flat[iLayer + 1] + h_flat[iLayer] + c_flat[iLayer]);
              else
                v_flat[iLayer + 1] = nrelu_mean(v_flat[iLayer + 1] + h_flat[iLayer] + c_flat[iLayer]);
            }
            cudaStreamSynchronize(0);
          }
          #pragma omp barrier

          for (int iLayer = min(observedLayer - 1, (int)cLayerCount); iLayer >= 0; --iLayer) {

            // If not the top-most layer, calculate convolutional top-down signal
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

                if (iMeanField == 0 && iLayer > 0)
                  cV[iLayer] = cV[iLayer] + *cF[iLayer][k] * repeat(ch[iLayer], cF[iLayer][k]->size() / ch[iLayer].size()) * 2.0;
                else
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
                if (iLayer == 0)
                  V_master[0] = (V_master[0] + b) * repeat(hMask[0], visSize[0] / layerSize[0]);

                v_master[iLayer] = rearrange_r(V_master[iLayer], rearrangeBlock[iLayer]);
                cudaStreamSynchronize(0);
              }
            } else {  // calculate top-down signal from first RBM
              #pragma omp master
              {
                v_flat[0] = prod(v_flat[1], tbblas::trans(W[0]));
                assert(v_flat[0].count() == v_master[iLayer].count());
                if (iMeanField == 0)
                  v_flat[0] = 2.0 * v_flat[0];
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
                h[iLayer - 1] = nrelu_mean(h[iLayer - 1]);
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
            }
            cudaStreamSynchronize(0);
            #pragma omp barrier
          }
        }

        #pragma omp master
        {
          if (queryLayer <= cLayerCount) {
            if (queryLayer == 0) {
              V_master[0] = (V_master[0] * dbm.getStddev() + dbm.getMean()) * repeat(hMask[0], visSize[0] / layerSize[0]);
              v_master[0] = rearrange_r(V_master[0], rearrangeBlock[0]);
            }
            outputs->push_back(boost::make_shared<host_tensor_t>(v_master[queryLayer]));
          } else {
            boost::shared_ptr<host_tensor_t> output(new host_tensor_t(1, 1, 1, v_flat[queryLayer - cLayerCount].count()));
            thrust::copy(v_flat[queryLayer - cLayerCount].begin(), v_flat[queryLayer - cLayerCount].end(), output->begin());
            outputs->push_back(output);
          }
        }
      }

      #pragma omp master
      if (monitor)
        monitor->reportProgress(100. * (i + 1) / inputs.size());
    }

    cudaStreamSynchronize(0);
    #pragma omp barrier

    // Free up memory
    for (size_t iLayer = 0; iLayer < cLayerCount; ++iLayer) {
      for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
        cF[iLayer][k] = cc[iLayer][k] = boost::shared_ptr<ctensor_t>();
      }
    }

    if (tid == 0) {
      for (int i = 1; i < gpuCount; ++i)
        cudaDeviceDisablePeerAccess(i);
    } else {
      cudaDeviceDisablePeerAccess(0);
    }

  } /* end of parallel */

  newState->setOutputs(outputs);
}

}

}

