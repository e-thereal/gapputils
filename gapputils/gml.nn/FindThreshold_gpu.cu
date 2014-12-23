/*
 * FindThreshold_gpu.cu
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#include "FindThreshold.h"

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/sequence_iterator.hpp>

#include "optlib/BrentOptimizer.h"

namespace gml {

namespace nn {

class Objective : public virtual optlib::IFunction<double> {
private:
  const v_host_tensor_t& labels;
  const std::vector<host_tensor_t>& predictions;

public:
  Objective(const v_host_tensor_t& labels, const std::vector<host_tensor_t>& predictions) : labels(labels), predictions(predictions) { }

  virtual double eval(const double& value) {
    using namespace tbblas;

    double DSC = 0;
    const float threshold = value;
    for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
      host_tensor_t& label = *labels[iSample];
      const host_tensor_t& pred = predictions[iSample];

      DSC += 2 * sum((label > 0.5) * (pred > threshold)) / (sum(label > 0.5) + sum(pred > threshold));
    }
    return DSC / labels.size();
  }
};

FindThresholdChecker::FindThresholdChecker() {
  FindThreshold test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(PatchWidth, test);
  CHECK_MEMORY_LAYOUT2(PatchHeight, test);
  CHECK_MEMORY_LAYOUT2(PatchDepth, test);
  CHECK_MEMORY_LAYOUT2(PatchCounts, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
}

void FindThreshold::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tensor_t::dim_t dim_t;

  v_host_tensor_t& data = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  dim_t inputSize = data[0]->size();
  dim_t patchSize = seq(getPatchWidth(), getPatchHeight(), getPatchDepth(), data[0]->size()[dimCount - 1]);
  dim_t range = data[0]->size() - patchSize + 1;
  dim_t patchCenter = patchSize / 2 * seq(1,1,1,0);
  dim_t patchCount = _PatchCounts.size() == 3 ? seq(_PatchCounts[0], _PatchCounts[1], _PatchCounts[2], 1) : seq<dimCount>(1);
  dim_t multiPatchCount = (range + patchCount - 1) / (patchCount);

  if (patchSize.prod() != getInitialModel()->visibles_count()) {
    dlog(Severity::Warning) << "Patch dimension doesn't match the number of visible units of the neural network. Aborting!";
    return;
  }

  if (data.size() != labels.size()) {
    dlog(Severity::Warning) << "Need to have same number of input samples and labels. Aborting!";
    return;
  }

  tbblas_print(range);
  tbblas_print(patchCount);
  tbblas_print(multiPatchCount);

  tbblas::deeplearn::nn<value_t> nn(*getInitialModel());
  nn.visibles().resize(seq(patchCount.prod(), patchSize.prod()));

  tensor_t sample, label, labelPatch;
  for (size_t iSample = 0; iSample < data.size() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
    sample = *data[iSample];
    label = zeros<value_t>(sample.size() * seq(1, 1, 1, 0) + seq(0, 0, 0, (int)getInitialModel()->hiddens_count()));

    for (sequence_iterator<dim_t> iSuperPatch(seq<dimCount>(0), multiPatchCount); iSuperPatch && (monitor ? !monitor->getAbortRequested() : true); ++iSuperPatch) {
      for (sequence_iterator<dim_t> iPatch(seq<dimCount>(0), patchCount); iPatch && (monitor ? !monitor->getAbortRequested() : true); ++iPatch) {
        if (*iSuperPatch * patchCount + *iPatch == min(*iSuperPatch * patchCount + *iPatch, range - 1))
          row(nn.visibles(), iPatch.current()) = reshape(sample[*iSuperPatch * patchCount + *iPatch, patchSize], 1, patchSize.prod());
        else
          row(nn.visibles(), iPatch.current()) = zeros<value_t>(1, patchSize.prod());
      }

      // Perform forward propagation
      nn.normalize_visibles();
      nn.infer_hiddens();

      dim_t overlap = min(patchCount, range - *iSuperPatch * patchCount);
      labelPatch = reshape(nn.hiddens(), patchCount);
      label[*iSuperPatch * patchCount + patchCenter, overlap] = labelPatch[seq<dimCount>(0), overlap];

      if (monitor) {
        monitor->reportProgress(100. * (iSuperPatch.current() + iSample * iSuperPatch.count()) / (iSuperPatch.count() * data.size()));
      }
    }

    predictions.push_back(label);
    tbblas::synchronize();
  }


  assert(data.size() == predictions.size());

  dlog(Severity::Message) << "Predictions calculated. Starting optimization...";

  Objective objective(labels, predictions);

  for (double t = 0; t <= 1; t += 0.05)
    dlog(Severity::Trace) << "DSC at " << t << " = " << objective.eval(t);

  optlib::BrentOptimizer optimizer;
  optimizer.setStepSize(0.2);
  optimizer.setTolerance(0.01);

  double threshold = 0.2;
  optimizer.maximize(threshold, objective);


  dlog(Severity::Message) << "Optimal threshold = " << threshold << " at a DSC of " << objective.eval(threshold);
}

}

}
