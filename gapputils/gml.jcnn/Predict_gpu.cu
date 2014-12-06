/*
 * Predict_gpu.cu
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#include "Predict.h"

#include <tbblas/deeplearn/joint_cnn.hpp>

namespace gml {

namespace jcnn {

PredictChecker::PredictChecker() {
  Predict test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(LeftInputs, test);
  CHECK_MEMORY_LAYOUT2(RightInputs, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void Predict::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  using namespace tbblas;

  typedef model_t::value_t value_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef tensor<value_t, dimCount, true> tensor_t;

  tbblas::deeplearn::joint_cnn<value_t, dimCount> cnn(*getModel());

  // Fill the visible units of the one layer NN
  v_host_tensor_t& lefts = *getLeftInputs();
  v_host_tensor_t& rights = *getRightInputs();
  boost::shared_ptr<v_data_t> outputs(new v_data_t());

  if (lefts.size() != rights.size()) {
    dlog(Severity::Warning) << "The left and right inputs must have the same number of tensors. Aborting!";
    return;
  }

  tensor_t left, right;
  int oldProgress = -1;
  for (size_t i = 0; i < lefts.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    left = *lefts[i];
    cnn.set_left_input(left);

    right = *rights[i];
    cnn.set_right_input(right);

    // Perform forward propagation
    cnn.normalize_visibles();
    cnn.infer_hiddens();

    // Collect results
    boost::shared_ptr<data_t> output(new data_t(getModel()->hiddens_count()));
    thrust::copy(cnn.hiddens().begin(), cnn.hiddens().end(), output->begin());
    outputs->push_back(output);

    if (monitor && oldProgress != (i + 1) * 100 / lefts.size()) {
      oldProgress = (i + 1) * 100 / lefts.size();
      monitor->reportProgress((i + 1) * 100. / lefts.size());
    }
  }
  tbblas::synchronize();

  newState->setOutputs(outputs);
}

}

}
