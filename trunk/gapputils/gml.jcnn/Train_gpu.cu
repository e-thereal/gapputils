/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/joint_cnn.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>

namespace gml {

namespace jcnn {

TrainChecker::TrainChecker() {
  Train test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(LeftTrainingSet, test);
  CHECK_MEMORY_LAYOUT2(RightTrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(LeftFilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(RightFilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(CLearningRate, test);
  CHECK_MEMORY_LAYOUT2(DLearningRate, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);
}

void Train::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;

  if (getLeftTrainingSet()->size() != getLabels()->size() || getRightTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

  tbblas::deeplearn::joint_cnn<value_t, dimCount> cnn(*model);
  for (size_t i = 0; i < model->left_cnn_layers().size() && i < getLeftFilterBatchSize().size(); ++i)
    cnn.set_left_batch_length(i, getLeftFilterBatchSize()[i]);
  for (size_t i = 0; i < model->right_cnn_layers().size() && i < getRightFilterBatchSize().size(); ++i)
    cnn.set_right_batch_length(i, getRightFilterBatchSize()[i]);

  // Prepare data
  v_host_tensor_t& leftData = *getLeftTrainingSet();
  v_host_tensor_t& rightData = *getRightTrainingSet();
  v_data_t& labels = *getLabels();

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  dlog() << "Preparation finished. Starting training.";

  const size_t batchSize = getBatchSize();
  const size_t batchCount = labels.size() / batchSize;

  value_t error;
  matrix_t target(1, model->hiddens_count());
  tensor_t left, right;

  for (int iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

      cnn.init_gradient_updates(getCLearningRate(), getDLearningRate(), momentum, weightcost);

      for (int iSample = 0; iSample < batchSize; ++iSample) {
        const int current = iSample + iBatch * batchSize;
        thrust::copy(labels[current]->begin(), labels[current]->end(), target.begin());
        left = *leftData[current];
        cnn.set_left_input(left);
        right = *rightData[current];
        cnn.set_right_input(right);
        cnn.normalize_visibles();
        cnn.infer_hiddens();

        error += sqrt(dot(cnn.hiddens() - target, cnn.hiddens() - target));

        cnn.update_gradient(target, getCLearningRate() / batchSize, getDLearningRate() / batchSize);
      }

      cnn.apply_gradient();
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / labels.size();

    if (monitor)
      monitor->reportProgress(100 * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
