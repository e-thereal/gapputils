/*
 * Predict_gpu.cu
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#include "Predict.h"

#include <tbblas/deeplearn/encoder.hpp>

namespace gml {

namespace encoder {

PredictChecker::PredictChecker() {
  Predict test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(SubRegionCount, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Predict::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef model_t::value_t value_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef tensor<value_t, dimCount, true> tensor_t;

  tbblas::deeplearn::encoder<value_t, dimCount> encoder(*getModel(), _SubRegionCount);
  for (size_t i = 0; i < getModel()->cnn_encoders().size() + getModel()->dnn_decoders().size() && i < getFilterBatchSize().size(); ++i)
    encoder.set_batch_length(i, getFilterBatchSize()[i]);

  // Fill the visible units of the one layer NN
  v_host_tensor_t& tensors = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

  tensor_t tensor;
  for (size_t i = 0; i < tensors.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    tensor = *tensors[i];
    encoder.inputs() = tensor;

    // Perform forward propagation
    encoder.infer_outputs();
    tensor = encoder.outputs();

    // Collect results
    outputs->push_back(boost::make_shared<host_tensor_t>(tensor));

    if (monitor) {
      monitor->reportProgress((i + 1) * 100. / tensors.size());
    }
  }
  tbblas::synchronize();

  newState->setOutputs(outputs);
}

}

}
