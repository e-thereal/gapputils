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

#include <omp.h>

#include "math.hpp"

namespace gml {

namespace convrbm4d {

InferenceChecker::InferenceChecker() {
  Inference test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
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

  dim_t size[layerCount], layerSize[layerCount];
  for (size_t iLayer = 0; iLayer < layerCount; ++iLayer) {
    size[iLayer] = layerSize[iLayer] = dbm.getHiddenBiases()->at(iLayer)->at(0)->size();
    size[iLayer] = dbm.getWeights()->at(iLayer)->at(0)->size()[dimCount - 1];
  }

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
  tensor_t output[layerCount], v_master[layerCount];
  ctensor_t cv_master[layerCount];

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

      // Copy filters to the device and pre-calculate the FFT
      {
        tensor_t f, h, kern, pad;
        ctensor_t cf, ch;
        for (size_t k = tid; k < filters[iLayer]->size(); k += gpuCount) {

          kern = *filters[iLayer]->at(k);
          dim_t topleft = size[iLayer] / 2 - kern.size() / 2;
          pad = zeros<value_t>(size[iLayer]);
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
}

}

}

