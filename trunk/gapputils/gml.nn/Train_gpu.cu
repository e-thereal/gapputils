/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/nn_layer.hpp>
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
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
}

void Train::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  typedef nn_layer_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  boost::shared_ptr<nn_layer_t> model(new nn_layer_t(*getInitialModel()));

  tbblas::deeplearn::nn_layer<value_t> nn_layer(*model);
  nn_layer.visibles().resize(seq((int)getBatchSize(), (int)model->visibles_count()));
  nn_layer.hiddens().resize(seq((int)getBatchSize(), (int)model->hiddens_count()));

  // Prepare data
  v_data_t& data = *getTrainingSet();
  v_data_t& labels = *getLabels();
  matrix_t X(data.size(), model->visibles_count());
  matrix_t Y(data.size(), model->hiddens_count());
  matrix_t yBatch(getBatchSize(), model->visibles_count());

  matrix_t res;

  for (size_t i = 0; i < data.size(); ++i) {
    thrust::copy(data[i]->begin(), data[i]->end(), row(X, i).begin());
    thrust::copy(labels[i]->begin(), labels[i]->end(), row(Y, i).begin());
  }

  {
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

  const size_t batchCount = data.size() / getBatchSize();

  value_t error;

  for (int iEpoch = 0; iEpoch < getEpochCount(); ++iEpoch) {

    error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {
      nn_layer.visibles() = X[seq(iBatch * getBatchSize(), 0), nn_layer.visibles().size()];
      yBatch = Y[seq(iBatch * getBatchSize(), 0), nn_layer.hiddens().size()];

      // Perform forward propagation
      nn_layer.normalize_visibles();
      nn_layer.infer_hiddens();
      error += sqrt(dot(nn_layer.hiddens() - yBatch, nn_layer.hiddens() - yBatch));

//      tbblas_print(dot(nn_layer.hiddens(), nn_layer.hiddens()));
//      tbblas_print(dot(model->weights(), model->weights()));
//      tbblas_print(dot(model->bias(), model->bias()));

      // Update model
      nn_layer.calculate_deltas(yBatch);
      nn_layer.update_model(getLearningRate(), momentum, weightcost);
    }

    dlog() << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / data.size();

    if (monitor)
      monitor->reportProgress(100 * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}


