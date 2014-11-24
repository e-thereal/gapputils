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
#include <tbblas/shift.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>
#include <tbblas/rearrange.hpp>

#include <omp.h>

#include <tbblas/deeplearn/math.hpp>
#include <tbblas/deeplearn/conv_rbm.hpp>
#include <tbblas/deeplearn/conv_rbm_model.hpp>

namespace gml {

namespace convrbm4d {

FilterChecker::FilterChecker() {
  Filter filter;
  filter.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, filter);
  CHECK_MEMORY_LAYOUT2(Inputs, filter);
  CHECK_MEMORY_LAYOUT2(Direction, filter);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, filter);
  CHECK_MEMORY_LAYOUT2(GpuCount, filter);
  CHECK_MEMORY_LAYOUT2(DoubleWeights, filter);
  CHECK_MEMORY_LAYOUT2(OnlyFilters, filter);
  CHECK_MEMORY_LAYOUT2(SampleUnits, filter);

  CHECK_MEMORY_LAYOUT2(Outputs, filter);
}

unsigned int upper_power_of_two(unsigned int v);

//#define TRACE std::cout << __LINE__ << std::endl;

void Filter::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  Logbook& dlog = getLogbook();
  model_t& model = *getModel();

  if (getFilterBatchSize() > model.filters().size() ||
      model.filters().size() % getFilterBatchSize() != 0)
  {
    dlog(Severity::Warning) << "Invalid FilterBatchSize. Aborting!";
    return;
  }

  std::vector<boost::shared_ptr<host_tensor_t> >& inputs = *getInputs();
  boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<host_tensor_t> >());

  conv_rbm<float, 4> crbm(model, getGpuCount());
  crbm.set_batch_length(getFilterBatchSize());

  tensor<float, 4, true> input;

  if (getDirection() == CodingDirection::Encode) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      input = *inputs[i];   // copies memory to the device. Makes rearranging faster
      crbm.visibles() = rearrange(input, model.stride_size());
      crbm.normalize_visibles();
      if (getSampleUnits())
        crbm.sample_hiddens();
      else
        crbm.infer_hiddens();
      outputs->push_back(boost::make_shared<host_tensor_t>(crbm.hiddens()));
      if (monitor)
        monitor->reportProgress(100. * i / inputs.size());
    }
  } else {
    for (size_t i = 0; i < inputs.size(); ++i) {
      crbm.hiddens() = *inputs[i];
      if (getSampleUnits())
        crbm.sample_visibles();
      else
        crbm.infer_visibles(getOnlyFilters());
      if (!getOnlyFilters())
        crbm.diversify_visibles();

      input = rearrange_r(crbm.visibles(), model.stride_size());
      outputs->push_back(boost::make_shared<host_tensor_t>(input));
      if (monitor)
        monitor->reportProgress(100. * i / inputs.size());
    }
  }

  newState->setOutputs(outputs);
}

}

}
