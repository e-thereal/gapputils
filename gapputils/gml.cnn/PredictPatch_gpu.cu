/*
 * PredictPatch_gpu.cu
 *
 *  Created on: Dec 4, 2014
 *      Author: tombr
 */

#include "PredictPatch.h"

#include <tbblas/deeplearn/cnn_patches.hpp>

namespace gml {

namespace cnn {

PredictPatchChecker::PredictPatchChecker() {
  PredictPatch test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(PatchCounts, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void PredictPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef model_t::value_t value_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  Logbook& dlog = getLogbook();
  dlog(Severity::Error) << "This module is not yet fully implemented.";
  if (true)
    return;

  // Fill the visible units of the one layer NN
  v_host_tensor_t& tensors = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

  // Set-up patch parameters
  dim_t patchCounts = (_PatchCounts.size() == 3 ? seq(_PatchCounts[0], _PatchCounts[1], _PatchCounts[2], 1) : seq<dimCount>(1));
  dim_t patchSize = getModel()->input_size() + patchCounts - 1;
  dim_t labelSize = patchCounts * seq(1, 1, 1, 0) + seq(0, 0, 0, (int)getModel()->hiddens_count());
  dim_t range = tensors[0]->size() - patchSize + 1;
  dim_t patchCenter = getModel()->input_size() / 2 * seq(1, 1, 1, 0);

  tbblas::deeplearn::cnn_patches<value_t, dimCount> cnn(*getModel(), patchCounts);

  tensor_t tensor, output;
  int oldProgress = -1;
  for (size_t i = 0; i < tensors.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    tensor = *tensors[i];

    // TODO: iterate over patches

    cnn.set_input(tensor);

    // Perform forward propagation
    cnn.normalize_visibles();
    cnn.infer_hiddens();

    // Collect results
//    boost::shared_ptr<data_t> output(new data_t(getModel()->hiddens_count()));
//    thrust::copy(cnn.hiddens().begin(), cnn.hiddens().end(), output->begin());
//    outputs->push_back(output);

    if (monitor && oldProgress != (i + 1) * 100 / tensors.size()) {
      oldProgress = (i + 1) * 100 / tensors.size();
      monitor->reportProgress((i + 1) * 100. / tensors.size());
    }
  }
  tbblas::synchronize();

  newState->setOutputs(outputs);
}

}

}
