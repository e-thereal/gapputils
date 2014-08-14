/*
 * Predict_gpu.cu
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#include "Predict.h"

#include <tbblas/deeplearn/nn_layer.hpp>

namespace gml {

namespace nn {

PredictChecker::PredictChecker() {
  Predict test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Predict::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef nn_layer_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  tbblas::deeplearn::nn_layer<value_t> nn_layer(*getModel());

  // Fill the visible units of the one layer NN
  v_data_t& data = *getInputs();
  nn_layer.visibles().resize(seq(data.size(), getModel()->visibles_count()));
  for (size_t i = 0; i < data.size(); ++i) {
    thrust::copy(data[i]->begin(), data[i]->end(), row(nn_layer.visibles(), i).begin());
  }

  // Perform forward propagation
  nn_layer.normalize_visibles();
  nn_layer.infer_hiddens();

  // Collect results
  boost::shared_ptr<v_data_t> outputs(new v_data_t());
  for (size_t i = 0; i < data.size(); ++i) {
    boost::shared_ptr<data_t> output(new data_t(getModel()->hiddens_count()));
    thrust::copy(row(nn_layer.hiddens(), i).begin(), row(nn_layer.hiddens(), i).end(), output->begin());
    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

}

}
