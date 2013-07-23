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

#include <omp.h>

#include "math.hpp"

namespace gml {

namespace convrbm4d {

InferenceChecker::InferenceChecker() {
  Inference test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Direction, test);
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

  // Get inputs
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();

  // Prepare outputs
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  // Load model into device memory
  DbmModel& dbm = *getModel();

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
  if (getDirection() == CodingDirection::Encode) {
    rearrangeBlock[0] = getInputs()->at(0)->size() / visSize[0];
    rearrangeBlock[0][dimCount - 1] = 1;
    assert(rearrangeBlock[0] == dbm.getVisibleBlockSize());
  } else {
    rearrangeBlock[0] = dbm.getVisibleBlockSize();
  }
  for (size_t iLayer = 1; iLayer < layerCount; ++iLayer)
    rearrangeBlock[iLayer] = dbm.getHiddenBiases()->at(iLayer - 1)->at(0)->size() / dbm.getHiddenBiases()->at(iLayer)->at(0)->size();

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

  vv_host_tensor_t& filters = *dbm.getWeights();
  vv_host_tensor_t& c = *dbm.getHiddenBiases();
  tensor_t b = *dbm.getVisibleBias();

  v_ctensor_t cF[layerCount], cc[layerCount];
  for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
    cF[iLayer].resize(filters[iLayer]->size());
    cc[iLayer].resize(filters[iLayer]->size());
  }
  tensor_t v_master[layerCount + 1], V_master[layerCount];
  ctensor_t cV_master[layerCount];

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

    for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
      hMask[iLayer] = *dbm.getMasks()->at(iLayer);

      // TODO: Copy filters to the device and pre-calculate the FFT
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

    tensor_t h[layerCount];
    ctensor_t cV[layerCount], ch_full[layerCount], ch[layerCount];
    tensor_t f[layerCount], kern[layerCount], pad[layerCount];
    ctensor_t cf[layerCount];

    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {

      if (getDirection() == CodingDirection::Encode) {

        /*** LOAD VISIBLE LAYER ***/

        cudaStreamSynchronize(0);
        #pragma omp barrier

        #pragma omp master
        {
          v_master[0] = *inputs[i];
          V_master[0] = rearrange(v_master[0], rearrangeBlock[0]);
          V_master[0] = (V_master[0] - dbm.getMean()) / dbm.getStddev();
          V_master[0] = V_master[0] * repeat(hMask[0], visSize[0] / layerSize[0]);
          cV_master[0] = fft(V_master[0], dimCount - 1, plan_v[0]);

          for (size_t iLayer = 0; iLayer < layerCount; ++iLayer)
            v_master[iLayer + 1] = zeros<value_t>(hidSize[iLayer]);
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        cV[0] = cV_master[0];

        // Perform multiple mean field updates (first update initialize the model)
        for (size_t iMeanField = 0; iMeanField < getIterations(); ++iMeanField) {
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
              }
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
        outputs->push_back(boost::make_shared<host_tensor_t>(v_master[layerCount]));
      } else { /*** DECODING ***/

        /*** LOAD HIDDEN LAYER ***/

        for (size_t iLayer = 0; iLayer < layerCount; ++iLayer)
          cV[iLayer] = zeros<complex_t>(cF[iLayer][0]->size(), cF[iLayer][0]->fullsize());

        #pragma omp master
        {
          v_master[layerCount] = *inputs[i];
          for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
            cV_master[iLayer] = zeros<complex_t>(cF[iLayer][0]->size(), cF[iLayer][0]->fullsize());
          }
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        // Perform multiple mean field updates (first update initialize the model)
        for (size_t iMeanField = 0; iMeanField < getIterations(); ++iMeanField) {

          for (int iLayer = layerCount - 1; iLayer >= 0; --iLayer) {

            // If not the top-most layer, calculate top-down signal
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
            }

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

        #pragma omp master
        {
          V_master[0] = (V_master[0] * dbm.getStddev() + dbm.getMean()) * repeat(hMask[0], visSize[0] / layerSize[0]);
          v_master[0] = rearrange_r(V_master[0], rearrangeBlock[0]);
          outputs->push_back(boost::make_shared<host_tensor_t>(v_master[0]));
        }
      }

      #pragma omp master
      if (monitor)
        monitor->reportProgress(100. * (i + 1) / inputs.size());
    }

    cudaStreamSynchronize(0);
    #pragma omp barrier

    // Free up memory
    for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
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

