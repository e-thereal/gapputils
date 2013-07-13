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
  CHECK_MEMORY_LAYOUT2(GpuCount, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

unsigned int upper_power_of_two(unsigned int v);

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
  rearrangeBlock[0] = getInputs()->at(0)->size() / visSize[0];
  rearrangeBlock[0][dimCount - 1] = 1;
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
  tensor_t input, output[layerCount], v_master[layerCount];
  ctensor_t cv_master[layerCount];

//  #pragma omp parallel
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

    tensor_t v[layerCount], h[layerCount];
    ctensor_t cv[layerCount], ch_full[layerCount], ch[layerCount];

    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {

      if (getDirection() == CodingDirection::Encode) {

        /*** LOAD VISIBLE LAYER ***/

        cudaStreamSynchronize(0);
        #pragma omp barrier

        #pragma omp master
        {
          input = *inputs[i];
          v_master[0] = rearrange(input, rearrangeBlock[0]);
          v_master[0] = (v_master[0] - dbm.getMean()) / dbm.getStddev();
          v_master[0] = v_master[0] * repeat(hMask[0], visSize[0] / layerSize[0]);
          cv_master[0] = fft(v_master[0], dimCount - 1, plan_v[0]);

          for (size_t iLayer = 0; iLayer < layerCount; ++iLayer)
            output[iLayer].resize(hidSize[iLayer], hidSize[iLayer]);
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        cv[0] = cv_master[0];

        bool validSize = true;
        for (unsigned j = 0; j < dimCount - 1; ++j) {
          if (v_master[0].size()[j] != upper_power_of_two(v_master[0].size()[j])) {
            dlog(Severity::Warning) << "The input size in each dimension must be a power of 2. Skipping image!";
            validSize = false;
            break;
          }
        }
        if (!validSize)
          continue;

        // Perform a single up pass to initialize the values
        for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
          for (size_t k = tid; k < cF[iLayer].size(); k += gpuCount) {
            ch_full[iLayer] = conj(*cF[iLayer][k]) * cv[iLayer];
            ch[iLayer] = sum(ch_full[iLayer], dimCount - 1);
            ch[iLayer] = ch[iLayer] + *cc[iLayer][k];
            h[iLayer] = ifft(ch[iLayer], dimCount - 1, iplan_h[iLayer]);
            h[iLayer] = nrelu_mean(h[iLayer]);

            output[iLayer][seq(0,0,0,(int)k), h[iLayer].size()] = h[iLayer] * hMask[iLayer];
          }
          cudaStreamSynchronize(0);
          #pragma omp barrier

          if (iLayer < layerCount - 1) {
            // rearrange into master first and then let all threads read from master into v
            #pragma omp master
            {
              v_master[iLayer + 1] = rearrange(output[iLayer], rearrangeBlock[iLayer + 1]);
              cv_master[iLayer + 1] = fft(v_master[iLayer + 1], dimCount - 1, plan_v[iLayer + 1]);
              cudaStreamSynchronize(0);
            }
            #pragma omp barrier
            cv[iLayer + 1] = cv_master[iLayer + 1];
          }
        }
        cudaStreamSynchronize(0);
        #pragma omp barrier

        // Perform multiple mean field updates

        #pragma omp master
        outputs->push_back(boost::make_shared<host_tensor_t>(output[layerCount - 1]));
      } else {

        // TODO: perform inference in the opposite direction

        if (i == 0)
          dlog(Severity::Warning) << "Decoding not yet implemented.";
      }

      #pragma omp master
      if (monitor)
        monitor->reportProgress(100. * i / inputs.size());
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

