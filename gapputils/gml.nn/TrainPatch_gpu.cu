/*
 * TrainPatch_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "TrainPatch.h"

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/reshape.hpp>

#include <tbblas/sequence_iterator.hpp>

namespace gml {

namespace nn {

TrainPatchChecker::TrainPatchChecker() {
  TrainPatch test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(Mask, test);
  CHECK_MEMORY_LAYOUT2(PatchWidth, test);
  CHECK_MEMORY_LAYOUT2(PatchHeight, test);
  CHECK_MEMORY_LAYOUT2(PatchDepth, test);
  CHECK_MEMORY_LAYOUT2(PatchCount, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(PositiveRatio, test);
  CHECK_MEMORY_LAYOUT2(Objective, test);
  CHECK_MEMORY_LAYOUT2(SensitivityRatio, test);
  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Patches, test);
  CHECK_MEMORY_LAYOUT2(Targets, test);
  CHECK_MEMORY_LAYOUT2(Predictions, test);
}

void TrainPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  const int dimCount = host_tensor_t::dimCount;

  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

  tbblas::deeplearn::nn<value_t> nn(*model);
  nn.set_objective_function(getObjective());
  nn.set_sensitivity_ratio(getSensitivityRatio());

  nn.visibles().resize(seq((int)getBatchSize() * getPatchCount(), (int)model->visibles_count()));

  // Prepare data
  v_host_tensor_t& data = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  if (getPatchWidth() * getPatchHeight() * getPatchDepth() * data[0]->size()[dimCount - 1] != model->visibles_count()) {
    dlog(Severity::Warning) << "Patch dimension doesn't match the number of visible units of the neural network. Aborting!";
    return;
  }

  if (getPositiveRatio() >= 0.0 && labels[0]->size()[dimCount - 1] != 1) {
    dlog(Severity::Warning) << "Positive ratio can only be used for binary classification (channels of the label image must be 1). Aborting!";
    return;
  }

  dim_t patchSize = seq(getPatchWidth(), getPatchHeight(), getPatchDepth(), data[0]->size()[dimCount - 1]);
  dim_t labelSize = seq(1, 1, 1, labels[0]->size()[dimCount - 1]);
  dim_t range = data[0]->size() - patchSize + 1;
  dim_t patchCenter = patchSize / 2 * seq(1,1,1,0);

  matrix_t yBatch(getBatchSize() * getPatchCount(), model->hiddens_count());

  matrix_t res;
  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  dlog() << "Preparation finished. Starting training.";

  const int batchSize = getBatchSize();
  const int batchCount = data.size() / batchSize;

  value_t error, PPV, DSC, TPR, TNR;
  tensor_t tensor, label;
  host_tensor_t h_label;

  boost::shared_ptr<v_host_tensor_t> patches(new v_host_tensor_t());
  newState->setPatches(patches);

  boost::shared_ptr<v_host_tensor_t> targets(new v_host_tensor_t());
  newState->setTargets(targets);

  boost::shared_ptr<v_host_tensor_t> predictions(new v_host_tensor_t());
  newState->setPredictions(predictions);

  std::vector<dim_t> positiveLocations, maskLocations;
  if (getMask()) {
    host_tensor_t& mask = *getMask();
    for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (mask.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
      if (mask[*pos + patchSize / 2 * seq(1, 1, 1, 0)] > 0) {   // The center is within the lesion mask
        maskLocations.push_back(*pos);
      }
    }
  }

  for (int iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    PPV = DSC = TPR = TNR = error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      for (int iSample = 0; iSample < batchSize; ++iSample) {
        // Fill batch with random patches
        tensor = *data[iBatch * batchSize + iSample];
        label = *labels[iBatch * batchSize + iSample];
        h_label = *labels[iBatch * batchSize + iSample];

        int lesionPatchCount = 0;

        if (getPositiveRatio() >= 0) {
          positiveLocations.clear();
          for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (label.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
            if (h_label[*pos + patchSize / 2 * seq(1, 1, 1, 0)] > 0) {   // The center is within a lesion
              positiveLocations.push_back(*pos);
            }
          }
        }

        for (int iPatch = 0; iPatch < getPatchCount(); ++iPatch) {
          dim_t topleft;
          if (positiveLocations.size() && (float)rand() / (float)RAND_MAX < getPositiveRatio())
            topleft = positiveLocations[rand() % positiveLocations.size()];
          else if (maskLocations.size())
            topleft = maskLocations[rand() % maskLocations.size()];
          else
            topleft = seq(rand() % range[0], rand() % range[1], rand() % range[2], 0);

          if (h_label[topleft + patchCenter] > 0.1)
            ++lesionPatchCount;

          row(nn.visibles(), iSample * getPatchCount() + iPatch) = reshape(tensor[topleft, patchSize], 1, model->visibles_count());
          row(yBatch, iSample * getPatchCount() + iPatch) = reshape(label[topleft + patchCenter, labelSize], 1, model->hiddens_count());

          if (iEpoch + 1 == getEpochCount() && iBatch + 1 == batchCount && iSample + 1 == batchSize) {
            patches->push_back(boost::make_shared<host_tensor_t>(tensor[topleft, patchSize]));
            targets->push_back(boost::make_shared<host_tensor_t>(label[topleft + patchCenter, labelSize]));
          }
        }
        if (iEpoch == 0 && iBatch == 0 && iSample == 0) {
          dlog(Severity::Trace) << "Lesion patches = " << lesionPatchCount << "; total number of patches = " << getPatchCount();
        }
      }

      // Perform forward propagation
      nn.normalize_visibles();
      nn.infer_hiddens();
      error += sqrt(dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch) / yBatch.size()[0]);

      if (iEpoch + 1 == getEpochCount() && iBatch + 1 == batchCount) {
        for (size_t iPatch = 0; iPatch < getPatchCount(); ++iPatch)
          predictions->push_back(boost::make_shared<host_tensor_t>(reshape(row(nn.hiddens(), (batchSize - 1) * getPatchCount() + iPatch), labelSize)));
      }

      PPV += sum((yBatch > 0.5) * (nn.hiddens() > 0.5)) / sum(nn.hiddens() > 0.5);
      DSC += 2 * sum ((yBatch > 0.5) * (nn.hiddens() > 0.5)) / (sum(yBatch > 0.5) + sum(nn.hiddens() > 0.5));
      TPR += sum((yBatch > 0.5) * (nn.hiddens() > 0.5)) / sum(yBatch > 0.5);
      TNR += sum((yBatch < 0.5) * (nn.hiddens() < 0.5)) / sum(yBatch < 0.5);

      // Update model
      switch (getMethod()) {
      case TrainingMethod::Momentum:
        nn.momentum_update(yBatch, getLearningRate(), momentum, weightcost);
        break;

      case TrainingMethod::AdaDelta:
        nn.adadelta_update(yBatch, getLearningRate(), momentum, weightcost);
        break;
      }
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / batchCount
        << " (PPV = " << PPV / batchCount << ", DSC = " << DSC / batchCount << ", TPR = " << TPR / batchCount << ", TNR = " << TNR / batchCount << ")";

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
