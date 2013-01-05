/*
 * Filter_gpu.cu
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "Filter.h"

#include <tbblas/fft.hpp>
#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/repeat.hpp>

#include <omp.h>

#include "math.hpp"

namespace gml {

namespace convrbm4d {

FilterChecker::FilterChecker() {
  Filter filter;
  filter.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, filter);
  CHECK_MEMORY_LAYOUT2(Inputs, filter);
  CHECK_MEMORY_LAYOUT2(Direction, filter);
  CHECK_MEMORY_LAYOUT2(GpuCount, filter);

  CHECK_MEMORY_LAYOUT2(Outputs, filter);
}

unsigned int upper_power_of_two(unsigned int v);

void Filter::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  const unsigned dimCount = Model::dimCount;
  typedef complex<value_t> complex_t;
  typedef fft_plan<dimCount> plan_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor<complex_t, dimCount, true> ctensor_t;
  typedef tensor<complex_t, dimCount, false> host_ctensor_t;
  typedef tensor_t::dim_t dim_t;

  // Get inputs
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();

  // Prepare outputs
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  // Load model into device memory
  Model& crbm = *getModel();

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

  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm.getFilters();
  std::vector<boost::shared_ptr<host_tensor_t> >& c = *crbm.getHiddenBiases();
  tensor_t b = *crbm.getVisibleBias();

  std::vector<boost::shared_ptr<ctensor_t> > cF(filters.size()), cc(filters.size());
  tensor_t output, v_master;
  ctensor_t cv_master;

  #pragma omp parallel
  {
    plan_t plan_v, iplan_v, plan_h, iplan_h;

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

    // Copy filters to the device and pre-calculate the FFT
    {
      tensor_t f, h;
      ctensor_t cf, ch;
      for (size_t k = tid; k < filters.size(); k += gpuCount) {
        f = *filters[k];
        cf = fft(f, dimCount - 1, plan_v);
        cF[k] = boost::make_shared<ctensor_t>(cf);

        h = *c[k];
        ch = fft(h, dimCount - 1, plan_h);
        cc[k] = boost::make_shared<ctensor_t>(ch);
      }
    }

    tensor_t v, h;
    ctensor_t cv, ch_full, ch;

    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {

      if (getDirection() == CodingDirection::Encode) {

        cudaStreamSynchronize(0);
        #pragma omp barrier

        #pragma omp master
        {
          v_master = *inputs[i];
          if (crbm.getVisibleUnitType() == UnitType::Gaussian)
            v_master = (v_master - crbm.getMean()) / crbm.getStddev();
          cv_master = fft(v_master, dimCount - 1, plan_v);
          output.resize(seq(v_master.size()[0], v_master.size()[1], v_master.size()[2], (int)cF.size()),
              seq(v_master.size()[0], v_master.size()[1], v_master.size()[2], (int)cF.size()));
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        cv = cv_master;

        bool validSize = true;
        for (unsigned j = 0; j < dimCount - 1; ++j) {
          if (v_master.size()[j] != upper_power_of_two(v_master.size()[j])) {
            dlog(Severity::Warning) << "The input size in each dimension must be a power of 2. Skipping image!";
            validSize = false;
            break;
          }
        }
        if (!validSize)
          continue;

        for (size_t k = tid; k < cF.size(); k += gpuCount) {
          ch_full = conj(*cF[k]) * cv;
          ch = sum(ch_full, dimCount - 1);
          ch = ch + *cc[k];
          h = ifft(ch, dimCount - 1, iplan_h);

          switch (crbm.getHiddenUnitType()) {
            case UnitType::Bernoulli: h = sigm(h); break;
            case UnitType::ReLU:      h = max(0.0, h);  break;
            case UnitType::MyReLU:    h = nrelu_mean(h); break;
            case UnitType::ReLU1:     h = min(1.0, max(0.0, h));  break;
            case UnitType::ReLU2:     h = min(2.0, max(0.0, h));  break;
            case UnitType::ReLU4:     h = min(4.0, max(0.0, h));  break;
            case UnitType::ReLU8:     h = min(8.0, max(0.0, h));  break;
            default:
              dlog(Severity::Warning) << "Unsupported hidden unit type: " << crbm.getVisibleUnitType();
          }
          output[seq(0,0,0,(int)k), h.size()] = h;
        }
        cudaStreamSynchronize(0);
        #pragma omp barrier

      } else {  /* getDirection() == Decoding */

        cv = zeros<complex_t>(cF[0]->size(), cF[0]->fullsize());

        #pragma omp master
        {
          cv_master = zeros<complex_t>(cF[0]->size(), cF[0]->fullsize());
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        for (size_t k = tid; k < cF.size(); k += gpuCount) {
          h = (*inputs[i])[seq(0,0,0,(int)k), seq(inputs[i]->size()[0],inputs[i]->size()[1], inputs[i]->size()[2],1)];
          ch = fft(h, dimCount - 1, plan_h);

          cv = cv + *cF[k] * repeat(ch, cF[k]->size() / ch.size());
        }

        #pragma omp critical
        {
          cv_master = cv_master + cv;
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier

        #pragma omp master
        {
          v = ifft(cv_master, dimCount - 1, iplan_v);

          switch(crbm.getVisibleUnitType()) {
            case UnitType::Bernoulli: v = sigm(v + b); break;
            case UnitType::Gaussian:  v = v + b;       break;
            default:
              dlog(Severity::Warning) << "Unsupported unit type: " << crbm.getVisibleUnitType();
          }
          output = (v * crbm.getStddev()) + crbm.getMean();
          cudaStreamSynchronize(0);
        }
        #pragma omp barrier
      }

      #pragma omp master
      {
        outputs->push_back(boost::make_shared<host_tensor_t>(output));
        if (monitor)
          monitor->reportProgress(100. * i / inputs.size());
      }
    }

    cudaStreamSynchronize(0);
    #pragma omp barrier

    // Free up memory
    for (size_t k = tid; k < cF.size(); k += gpuCount) {
      cF[k] = cc[k] = boost::shared_ptr<ctensor_t>();
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
