/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/cnn.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>

namespace gml {

namespace cnn {

TrainChecker::TrainChecker() {
  Train test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
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

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

  tbblas::deeplearn::cnn<value_t, dimCount> cnn(*model);
  for (size_t i = 0; i < model->cnn_layers().size() && i < getFilterBatchSize().size(); ++i)
    cnn.set_batch_length(i, getFilterBatchSize()[i]);

  // Prepare data
  v_host_tensor_t& tensors = *getTrainingSet();
  v_data_t& labels = *getLabels();

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  dlog() << "Preparation finished. Starting training.";

  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;

  value_t error;
  matrix_t target(1, model->hiddens_count());
  tensor_t v;

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
        v = *tensors[current];
        cnn.set_input(v);
        cnn.normalize_visibles();
        cnn.infer_hiddens();

        error += sqrt(dot(cnn.hiddens() - target, cnn.hiddens() - target));

        cnn.update_gradient(target, getCLearningRate() / batchSize, getDLearningRate() / batchSize);
      }

      cnn.apply_gradient();
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / tensors.size();

    if (monitor)
      monitor->reportProgress(100 * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
