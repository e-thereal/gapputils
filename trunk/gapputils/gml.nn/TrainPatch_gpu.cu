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

namespace gml {

namespace nn {

TrainPatchChecker::TrainPatchChecker() {
  TrainPatch test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(PatchWidth, test);
  CHECK_MEMORY_LAYOUT2(PatchHeight, test);
  CHECK_MEMORY_LAYOUT2(PatchDepth, test);
  CHECK_MEMORY_LAYOUT2(PatchCount, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(BatchedLearning, test);
  CHECK_MEMORY_LAYOUT2(EqualizeClasses, test);
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Patches, test);
}

void TrainPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  const int dimCount = host_tensor_t::dimCount;

  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

  tbblas::deeplearn::nn<value_t> nn(*model);
  if (getBatchedLearning())
    nn.visibles().resize(seq((int)getBatchSize() * getPatchCount(), (int)model->visibles_count()));
  else
    nn.visibles().resize(seq(1, (int)model->visibles_count()));

  // Prepare data
  v_host_tensor_t& data = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  if (getPatchWidth() * getPatchHeight() * getPatchDepth() * data[0]->size()[dimCount - 1] != model->visibles_count()) {
    dlog(Severity::Warning) << "Patch dimension doesn't match the number of visible units of the neural network. Aborting!";
    return;
  }

  if (getEqualizeClasses() && labels[0]->size()[dimCount - 1] != 1) {
    dlog(Severity::Warning) << "Class equalization can only be used for binary classification (channels of the label image must be 1). Aborting!";
    return;
  }

  dim_t patchSize = seq(getPatchWidth(), getPatchHeight(), getPatchDepth(), data[0]->size()[dimCount - 1]);
  dim_t labelSize = seq(1, 1, 1, labels[0]->size()[dimCount - 1]);
  dim_t range = data[0]->size() - patchSize;
  range[dimCount - 1] = 1;
  dim_t patchCenter = patchSize / 2;
  patchCenter[dimCount - 1] = 0;

  matrix_t yBatch(getBatchedLearning() ? getBatchSize() * getPatchCount() : 1, model->hiddens_count());

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

  for (int iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    PPV = DSC = TPR = TNR = error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      if (getBatchedLearning()) {

        for (int iSample = 0; iSample < batchSize; ++iSample) {
          // Fill batch with random patches
          tensor = *data[iBatch * batchSize + iSample];
          label = *labels[iBatch * batchSize + iSample];
          h_label = *labels[iBatch * batchSize + iSample];

          // get number of
          value_t ratio = 0;
          if (getEqualizeClasses()) {
            value_t positives = sum(label[patchCenter, range]);
            value_t totals = label[patchCenter, range].count();
            value_t negatives = totals - positives;
            ratio = positives / negatives;
          }

          int lesionPatchCount = 0;

          for (int iPatch = 0; iPatch < getPatchCount(); ++iPatch) {
            dim_t topleft = seq(rand() % (range[0] + 1), rand() % (range[1] + 1), rand() % (range[2] + 1), 0);

            while (getEqualizeClasses() && h_label[topleft + patchCenter] < 0.1 && ((value_t)rand() / (value_t)RAND_MAX) > ratio) {
              topleft = seq(rand() % (range[0] + 1), rand() % (range[1] + 1), rand() % (range[2] + 1), 0);
            }

            if (label[topleft + patchCenter] > 0.1)
              ++lesionPatchCount;

            thrust::copy(tensor[topleft, patchSize].begin(),              tensor[topleft, patchSize].end(),              row(nn.visibles(), iSample * getPatchCount() + iPatch).begin());
            thrust::copy(label[topleft + patchCenter, labelSize].begin(), label[topleft + patchCenter, labelSize].end(), row(yBatch, iSample * getPatchCount() + iPatch).begin());

            if (iEpoch == 0 && iBatch == 0 && iSample == 0) {
              patches->push_back(boost::make_shared<host_tensor_t>(tensor[topleft, patchSize]));
            }
          }
          if (iEpoch == 0 && iBatch == 0 && iSample == 0) {
            dlog(Severity::Trace) << "Lesion ratio: " << ratio;
            dlog(Severity::Trace) << "Lesion patches = " << lesionPatchCount << "; total number of patches = " << getPatchCount();
          }
        }

//        nn.visibles() = X[seq(iBatch * getBatchSize(), 0), nn.visibles().size()];
//        yBatch = Y[seq(iBatch * getBatchSize(), 0), yBatch.size()];

        // Perform forward propagation
        nn.normalize_visibles();
        nn.infer_hiddens();
        error += sqrt(dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch) / yBatch.size()[0]);

        PPV += sum((yBatch > 0.5) * (nn.hiddens() > 0.5)) / sum(nn.hiddens() > 0.5);
        DSC += 2 * sum ((yBatch > 0.5) * (nn.hiddens() > 0.5)) / (sum(yBatch > 0.5) + sum(nn.hiddens() > 0.5));
        TPR += sum((yBatch > 0.5) * (nn.hiddens() > 0.5)) / sum(yBatch > 0.5);
        TNR += sum((yBatch < 0.5) * (nn.hiddens() < 0.5)) / sum(yBatch < 0.5);

        // Update model
        nn.update_model(yBatch, getLearningRate(), momentum, weightcost);
      } else {

//        nn.init_gradient_updates(getLearningRate() / batchSize, momentum, weightcost);
//
//        for (int iSample = 0; iSample < batchSize; ++iSample) {
//          nn.visibles() = X[seq(iBatch * batchSize + iSample, 0), nn.visibles().size()];
//          yBatch = Y[seq(iBatch * batchSize + iSample, 0), yBatch.size()];
//
//          // Perform forward propagation
//          nn.normalize_visibles();
//          nn.infer_hiddens();
//          error += sqrt(dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch));
//
//          // Update model
//          nn.update_gradient(yBatch, getLearningRate() / batchSize);
//        }
//
//        nn.apply_gradient();
      }
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / data.size()
        << " (PPV = " << PPV / batchCount << ", DSC = " << DSC / batchCount << ", TPR = " << TPR / batchCount << ", TNR = " << TNR / batchCount << ")";

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
