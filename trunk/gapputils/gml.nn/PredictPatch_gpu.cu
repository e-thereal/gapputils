/*
 * PredictPatch_gpu.cu
 *
 *  Created on: Dec 06, 2014
 *      Author: tombr
 */

#include "PredictPatch.h"

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/sequence_iterator.hpp>

namespace gml {

namespace nn {

PredictPatchChecker::PredictPatchChecker() {
  PredictPatch test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Inputs, test);
  CHECK_MEMORY_LAYOUT2(PatchWidth, test);
  CHECK_MEMORY_LAYOUT2(PatchHeight, test);
  CHECK_MEMORY_LAYOUT2(PatchDepth, test);
  CHECK_MEMORY_LAYOUT2(PatchCounts, test);
  CHECK_MEMORY_LAYOUT2(Outputs, test);
}

void PredictPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  const int dimCount = host_tensor_t::dimCount;
  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  v_host_tensor_t& data = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

  dim_t inputSize = data[0]->size();
  dim_t patchSize = seq(getPatchWidth(), getPatchHeight(), getPatchDepth(), data[0]->size()[dimCount - 1]);
  dim_t range = data[0]->size() - patchSize + 1;
  dim_t patchCenter = patchSize / 2 * seq(1,1,1,0);
  dim_t patchCount = _PatchCounts.size() == 3 ? seq(_PatchCounts[0], _PatchCounts[1], _PatchCounts[2], 1) : seq<dimCount>(1);
  dim_t multiPatchCount = (range + patchCount - 1) / (patchCount);

  if (patchSize.prod() != getModel()->visibles_count()) {
    dlog(Severity::Warning) << "Patch dimension doesn't match the number of visible units of the neural network. Aborting!";
    return;
  }

  tbblas_print(range);
  tbblas_print(patchCount);
  tbblas_print(multiPatchCount);

  tbblas::deeplearn::nn<value_t> nn(*getModel());
  nn.visibles().resize(seq(patchCount.prod(), patchSize.prod()));

  tensor_t sample, label, labelPatch;
  for (size_t iSample = 0; iSample < data.size() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
    sample = *data[iSample];
    label = zeros<value_t>(sample.size() * seq(1, 1, 1, 0) + seq(0, 0, 0, (int)getModel()->hiddens_count()));

    for (sequence_iterator<dim_t> iSuperPatch(seq<dimCount>(0), multiPatchCount); iSuperPatch && (monitor ? !monitor->getAbortRequested() : true); ++iSuperPatch) {
      for (sequence_iterator<dim_t> iPatch(seq<dimCount>(0), patchCount); iPatch && (monitor ? !monitor->getAbortRequested() : true); ++iPatch) {
        if (*iSuperPatch * patchCount + *iPatch == min(*iSuperPatch * patchCount + *iPatch, range - 1))
          row(nn.visibles(), iPatch.current()) = reshape(sample[*iSuperPatch * patchCount + *iPatch, patchSize], 1, patchSize.prod());
        else
          row(nn.visibles(), iPatch.current()) = zeros<value_t>(1, patchSize.prod());
      }

      // Perform forward propagation
      nn.infer_hiddens();

      dim_t overlap = min(patchCount, range - *iSuperPatch * patchCount);
      labelPatch = reshape(nn.hiddens(), patchCount);
      label[*iSuperPatch * patchCount + patchCenter, overlap] = labelPatch[seq<dimCount>(0), overlap];

      if (monitor) {
        monitor->reportProgress(100. * (iSuperPatch.current() + iSample * iSuperPatch.count()) / (iSuperPatch.count() * data.size()));
      }
    }

    outputs->push_back(boost::make_shared<host_tensor_t>(label));
    tbblas::synchronize();
  }

  newState->setOutputs(outputs);
}

}

}
