/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>

namespace gml {

namespace nn {

TrainChecker::TrainChecker() {
  Train test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(BatchedLearning, test);

  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(WeightCosts, test);
  CHECK_MEMORY_LAYOUT2(ShuffleTrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
}

void Train::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

  tbblas::deeplearn::nn<value_t> nn(*model);
  if (getBatchedLearning())
    nn.visibles().resize(seq((int)getBatchSize(), (int)model->visibles_count()));
  else
    nn.visibles().resize(seq(1, (int)model->visibles_count()));

  // Prepare data
  v_data_t& data = *getTrainingSet();
  v_data_t& labels = *getLabels();
  matrix_t X(data.size(), model->visibles_count());
  matrix_t Y(data.size(), model->hiddens_count());

  matrix_t yBatch(getBatchedLearning() ? getBatchSize() : 1, model->hiddens_count());

  matrix_t res;

  host_matrix_t h_X(data.size(), model->visibles_count());
  host_matrix_t h_Y(data.size(), model->hiddens_count());

  for (size_t i = 0; i < data.size(); ++i) {
    thrust::copy(data[i]->begin(), data[i]->end(), row(h_X, i).begin());
    thrust::copy(labels[i]->begin(), labels[i]->end(), row(h_Y, i).begin());
  }

  dlog() << "Data copied to the CPU.";

  X = h_X;
  Y = h_Y;

  dlog() << "Data copied to the GPU.";

  if (getShuffleTrainingSet()) {
    matrix_t trow, lrow;
    for (unsigned i = X.size()[0] - 1; i > 0; --i) {
      unsigned j = rand() % (i + 1);
      trow = row(X, i);
      row(X, i) = row(X, j);
      row(X, j) = trow;

      lrow = row(Y, i);
      row(Y, i) = row(Y, j);
      row(Y, j) = lrow;
    }
  }

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  dlog() << "Preparation finished. Starting training.";

  const int batchSize = getBatchSize();
  const int batchCount = data.size() / batchSize;

  value_t error;

  for (int iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      if (getBatchedLearning()) {
        nn.visibles() = X[seq(iBatch * getBatchSize(), 0), nn.visibles().size()];
        yBatch = Y[seq(iBatch * batchSize, 0), yBatch.size()];

        // Perform forward propagation
        nn.normalize_visibles();

        // Update model
        switch (getMethod()) {
        case TrainingMethod::Momentum:
          nn.momentum_update(yBatch, getLearningRate(), momentum, weightcost);
          break;

        case TrainingMethod::AdaDelta:
          nn.adadelta_update(yBatch, getLearningRate(), 0.95, weightcost);
          break;
        }

        error += dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch);
      } else {

        for (int iSample = 0; iSample < batchSize; ++iSample) {
          nn.visibles() = X[seq(iBatch * batchSize + iSample, 0), nn.visibles().size()];
          yBatch = Y[seq(iBatch * batchSize + iSample, 0), yBatch.size()];

          // Perform forward propagation
          nn.normalize_visibles();
          nn.infer_hiddens();
          error += dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch);

          // Update model
          nn.update_gradient(yBatch);
        }

        switch (getMethod()) {
        case TrainingMethod::Momentum:
          nn.momentum_step(getLearningRate(), momentum, weightcost);
          break;

        case TrainingMethod::AdaDelta:
          nn.adadelta_step(getLearningRate(), 0.95, weightcost);
          break;
        }
      }
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << sqrt(error / data.size());

    if (monitor)
      monitor->reportProgress(100 * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
