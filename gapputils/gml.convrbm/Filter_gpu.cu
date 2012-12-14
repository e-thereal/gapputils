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

#include "math.hpp"

namespace gml {

namespace convrbm {

FilterChecker::FilterChecker() {
  Filter filter;
  filter.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, filter);
  CHECK_MEMORY_LAYOUT2(Inputs, filter);
  CHECK_MEMORY_LAYOUT2(Direction, filter);

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
  typedef tensor_t::dim_t dim_t;

  // Get inputs
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();

  // Prepare outputs
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  // Load model into device memory
  Model& crbm = *getModel();
  plan_t plan_v, iplan_v, plan_h, iplan_h;

  // Copy filters to the device and pre-calculate the FFT
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm.getFilters();
  std::vector<ctensor_t > cF;
  {
    tensor_t f;
    ctensor_t cf;
    for (size_t i = 0; i < filters.size(); ++i) {
      f = *filters[i];
      cf = fft(f, dimCount - 1, plan_v);
      cF.push_back(cf);
    }
  }

  tensor_t b = *crbm.getVisibleBias();

  std::vector<boost::shared_ptr<host_tensor_t> >& c = *crbm.getHiddenBiases();
  std::vector<ctensor_t> cc;
  {
    tensor_t h;
    ctensor_t ch;
    for (size_t i = 0; i < c.size(); ++i) {
      h = *c[i];
      ch = fft(h, dimCount - 1, plan_h);
      cc.push_back(ch);
    }
  }

  tensor_t v, h;
  ctensor_t cv, ch_full, ch;

  for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    boost::shared_ptr<host_tensor_t> output(new host_tensor_t());

    if (getDirection() == CodingDirection::Encode) {
      v = *inputs[i];

      for (unsigned j = 0; j < dimCount - 1; ++j) {
        if (v.size()[j] != upper_power_of_two(v.size()[j])) {
          dlog(Severity::Warning) << "The input size in each dimension must be a power of 2. Aborting!";
          return;
        }
      }

      if (crbm.getVisibleUnitType() == UnitType::Gaussian)
        v = (v - crbm.getMean()) / crbm.getStddev();
      cv = fft(v, dimCount - 1, plan_v);
      output->resize(seq(v.size()[0], v.size()[1], (int)cF.size()), seq(v.size()[0], v.size()[1], (int)cF.size()));

      for (size_t k = 0; k < cF.size(); ++k) {
        ch_full = conj(cF[k]) * cv;
        ch = sum(ch_full, dimCount - 1);
        ch = ch + cc[k];
        h = ifft(ch, iplan_h);

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
        (*output)[seq(0,0,(int)k), h.size()] = h;
      }
    } else {  /* getDirection() == Decoding */

      cv = zeros<complex_t>(cF[0].size(), cF[0].fullsize());

      for (size_t k = 0; k < cF.size(); ++k) {
        h = (*inputs[i])[seq(0,0,(int)k), seq(inputs[i]->size()[0],inputs[i]->size()[1],1)];
        ch = fft(h, plan_h);

        cv = cv + cF[k] * repeat(ch, cF[k].size() / ch.size());
      }
      v = ifft(cv, dimCount - 1, iplan_v);

      switch(crbm.getVisibleUnitType()) {
        case UnitType::Bernoulli: v = sigm(v + b); break;
        case UnitType::Gaussian:  v = v + b;       break;
        default:
          dlog(Severity::Warning) << "Unsupported unit type: " << crbm.getVisibleUnitType();
      }
      *output = (v * crbm.getStddev()) + crbm.getMean();
    }

    outputs->push_back(output);
    if (monitor)
      monitor->reportProgress(100. * i / inputs.size());
  }

  newState->setOutputs(outputs);
}

}

}
