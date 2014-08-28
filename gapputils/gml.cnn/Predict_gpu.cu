/*
 * Predict_gpu.cu
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#include "Predict.h"

#include <tbblas/deeplearn/cnn.hpp>

namespace gml {

namespace cnn {

PredictChecker::PredictChecker() {
  Predict test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
  CHECK_MEMORY_LAYOUT2(FirstLayer, test);
}

void Predict::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef model_t::value_t value_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef tensor<value_t, dimCount, true> tensor_t;

  tbblas::deeplearn::cnn<value_t, dimCount> cnn(*getModel());

  // Fill the visible units of the one layer NN
  v_host_tensor_t& tensors = *getInputs();
  boost::shared_ptr<v_data_t> outputs(new v_data_t());
  boost::shared_ptr<v_host_tensor_t> firstLayer(new v_host_tensor_t());

  tbblas::deeplearn::cnn_layer<value_t, dimCount> cnn_layer(*getModel()->cnn_layers()[0]);

  tensor_t tensor;
  int oldProgress = -1;
  for (size_t i = 0; i < tensors.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    tensor = *tensors[i];
    cnn.set_input(tensor);

    // Perform forward propagation
    cnn.normalize_visibles();
    cnn.infer_hiddens();

    // Collect results
    boost::shared_ptr<data_t> output(new data_t(getModel()->hiddens_count()));
    thrust::copy(cnn.hiddens().begin(), cnn.hiddens().end(), output->begin());
    outputs->push_back(output);

    cnn_layer.visibles() = rearrange(tensor, getModel()->cnn_layers()[0]->stride_size());
    cnn_layer.normalize_visibles();
    cnn_layer.infer_hiddens();

    firstLayer->push_back(boost::make_shared<host_tensor_t>(cnn_layer.hiddens()));

    if (monitor && oldProgress != (i + 1) * 100 / tensors.size()) {
      oldProgress = (i + 1) * 100 / tensors.size();
      monitor->reportProgress((i + 1) * 100. / tensors.size());
    }
  }
  tbblas::synchronize();

  newState->setOutputs(outputs);
  newState->setFirstLayer(firstLayer);
}

}

}
