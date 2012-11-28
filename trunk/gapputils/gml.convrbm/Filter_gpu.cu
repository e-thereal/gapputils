/*
 * Filter_gpu.cu
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "Filter.h"

#include <tbblas/math.hpp>
#include <tbblas/zeros.hpp>

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

void Filter::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  const unsigned dimCount = Model::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  // Get inputs
  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();

  // Prepare outputs
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  // Load model into device memory
  Model& crbm = *getModel();
  std::vector<boost::shared_ptr<host_tensor_t> >& filters = *crbm.getFilters();
  std::vector<tensor_t > F;
  for (size_t i = 0; i < filters.size(); ++i) {
    tensor_t filter(filters[i]->size());
    thrust::copy(filters[i]->begin(), filters[i]->end(), filter.begin());
    F.push_back(filter);
  }
  value_t b = crbm.getVisibleBias();
  std::vector<value_t>& c = *crbm.getHiddenBiases();

  value_t mean = crbm.getMean(), sd = crbm.getStddev();

  tensor_t v, vtemp, h, padded;

  for (size_t i = 0; i < inputs.size(); ++i) {
    host_tensor_t& input = *inputs[i];
    boost::shared_ptr<host_tensor_t> output(new host_tensor_t());

    if (getDirection() == CodingDirection::Encode) {

      v.resize(input.size(), input.size());
      thrust::copy(input.begin(), input.end(), v.begin());
      v = (v - mean) / sd;

      dim_t hiddenSize = v.size() - F[0].size() + 1;
      dim_t outSize = hiddenSize;
      outSize[dimCount - 1] = F.size();

      dim_t start(0);
      *output = zeros<value_t>(outSize);
      for (size_t k = 0; k < F.size(); ++k) {
        h = conv(flip(F[k]), v);
        h = sigm(h + c[k]);

        start[dimCount - 1] = k;
        (*output)[start, hiddenSize] = h;
      }
    } else {
      h.resize(input.size(), input.size());
      thrust::copy(input.begin(), input.end(), h.begin());

      dim_t hiddenSize = h.size();
      hiddenSize[dimCount - 1] = 1;
      dim_t visibleSize = hiddenSize + F[0].size() - 1;
      dim_t paddedSize = visibleSize + F[0].size() - 1;
      dim_t start(0);

      v = zeros<value_t>(visibleSize);
      padded.resize(paddedSize, paddedSize);

      for (size_t k = 0; k < F.size(); ++k) {
        start[dimCount - 1] = k;
        padded[F[0].size() - 1, hiddenSize] = h[start, hiddenSize];
        vtemp = conv(F[k], padded);
        v = v + vtemp;
      }

      switch(crbm.getVisibleUnitType()) {
        case UnitType::Bernoulli: v = sigm(v + b); break;
        case UnitType::Gaussian:  v = v + b;       break;
        default:
          dlog(Severity::Warning) << "Unsupported unit type: " << crbm.getVisibleUnitType();
      }
      *output = (v * sd) + mean;
    }

    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

}

}
