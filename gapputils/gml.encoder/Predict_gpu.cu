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
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(MaximumLayer, test);
  CHECK_MEMORY_LAYOUT2(CalculateDeltas, test);
  CHECK_MEMORY_LAYOUT2(Objective, test);
  CHECK_MEMORY_LAYOUT2(SensitivityRatio, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(SubRegionCount, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Predict::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef model_t::value_t value_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef tensor<value_t, dimCount, true> tensor_t;


  tbblas::deeplearn::encoder<value_t, dimCount> encoder(*getModel(), _SubRegionCount);
  for (size_t i = 0; i < getModel()->cnn_encoders().size() + getModel()->dnn_decoders().size() && i < getFilterBatchSize().size(); ++i)
    encoder.set_batch_length(i, getFilterBatchSize()[i]);

  v_host_tensor_t& tensors = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

  if (_CalculateDeltas) {
    if (!_Labels || _Labels->size() != tensors.size()) {
      dlog(Severity::Warning) << "Labels are required for calculating deltas. Aborting!";
      return;
    }
  }

  tensor_t tensor, output, target;
  for (size_t i = 0; i < tensors.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    tensor = *tensors[i];

    encoder.inputs() = tensor;

    if (_CalculateDeltas) {
      target = *_Labels->at(i);
      encoder.infer_deltas(getMaximumLayer() > -1 ? getMaximumLayer() : getModel()->layer_count(), target);
    } else {
      encoder.infer_layer(getMaximumLayer() > -1 ? getMaximumLayer() : getModel()->layer_count());
    }

    output = encoder.outputs();

    // Collect results
    outputs->push_back(boost::make_shared<host_tensor_t>(output));

    if (monitor) {
      monitor->reportProgress((i + 1) * 100. / tensors.size());
    }
  }
  tbblas::synchronize();

  newState->setOutputs(outputs);
}

}

}
