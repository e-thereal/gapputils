/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/deeplearn/nn_base.hpp>

#include <tbblas/deeplearn/opt/classic_momentum.hpp>
#include <tbblas/deeplearn/opt/adadelta.hpp>
#include <tbblas/deeplearn/opt/adam.hpp>
#include <tbblas/deeplearn/opt/adam2.hpp>
#include <tbblas/deeplearn/opt/hessian_free2.hpp>

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

namespace td = tbblas::deeplearn;

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

  typedef td::nn_base<value_t> nn_base_t;
  typedef td::nn<value_t, td::opt::classic_momentum<value_t> > cm_nn_t;
  typedef td::nn<value_t, td::opt::adadelta<value_t> > adadelta_nn_t;
  typedef td::nn<value_t, td::opt::adam<value_t> > adam_nn_t;
  typedef td::nn<value_t, td::opt::adam2<value_t> > adam2_nn_t;
  typedef td::nn<value_t> default_nn_t;

//  boost::shared_ptr<nn_base_t> p_nn;

//  switch (getMethod()) {
//  case TrainingMethod::ClassicMomentum:
//    p_nn = boost::make_shared<cm_nn_t>(boost::ref(*model));
//    break;
//
//  case TrainingMethod::AdaDelta:
//    p_nn = boost::make_shared<adadelta_nn_t>(boost::ref(*model));
//    break;
//
//  case TrainingMethod::Adam:
//    p_nn = boost::make_shared<adam_nn_t>(boost::ref(*model));
//    break;
//
//  case TrainingMethod::AdamDecay:
//    p_nn = boost::make_shared<adam2_nn_t>(boost::ref(*model));
//    break;
//
//  default:
//    p_nn = boost::make_shared<default_nn_t>(boost::ref(*model));
//  }

//  nn_base_t& nn = *p_nn;

  cm_nn_t nn(*model);
  nn.set_learning_rate(getLearningRate());
  td::opt::hessian_free2<cm_nn_t> trainer(nn);
  trainer.set_weightcost(getWeightCosts());
  trainer.set_iteration_count(10);

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

  matrix_t input, label;

  for (int iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      if (getBatchedLearning()) {

        input = X[seq(iBatch * getBatchSize(), 0), nn.visibles().size()];
        label = Y[seq(iBatch * batchSize, 0), yBatch.size()];
        trainer.check_gradient(input, label, getLearningRate());
        trainer.check_Gv(input, label, getLearningRate(), getEpochCount(), getShuffleTrainingSet());
        return;

        nn.visibles() = X[seq(iBatch * getBatchSize(), 0), nn.visibles().size()];
        yBatch = Y[seq(iBatch * batchSize, 0), yBatch.size()];

        // Perform forward propagation

//        // Update model
//        switch (getMethod()) {
//        case TrainingMethod::ClassicMomentum:
//          {
//            boost::shared_ptr<cm_nn_t> cm_nn = boost::dynamic_pointer_cast<cm_nn_t>(p_nn);
//            cm_nn->set_learning_rate(getLearningRate());
//            cm_nn->set_momentum(momentum);
//          }
//          break;
//
//        case TrainingMethod::AdaDelta:
//          {
//            boost::shared_ptr<adadelta_nn_t> ad_nn = boost::dynamic_pointer_cast<adadelta_nn_t>(p_nn);
//            ad_nn->set_epsilon(getLearningRate());
//            ad_nn->set_decay_rate(0.95);
//          }
//          break;
//
//        default:
//          dlog(Severity::Warning) << "Training method " << getMethod() << " has not been implemented.";
//        }
        nn.set_momentum(momentum);
        nn.update(yBatch, weightcost);

//        error += dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch) / model->hiddens_count();

        error += nn.loss(yBatch);
//        tbblas_print(nn.loss(yBatch));
      } else {

//        for (int iSample = 0; iSample < batchSize; ++iSample) {
//          nn.visibles() = X[seq(iBatch * batchSize + iSample, 0), nn.visibles().size()];
//          yBatch = Y[seq(iBatch * batchSize + iSample, 0), yBatch.size()];
//
//          // Perform forward propagation
//          nn.normalize_visibles();
//          nn.infer_hiddens();
//          error += dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch);
//
//          // Update model
//          nn.update_gradient(yBatch);
//        }
//
//        switch (getMethod()) {
//        case TrainingMethod::ClassicMomentum:
//          {
//            boost::shared_ptr<cm_nn_t> cm_nn = boost::dynamic_pointer_cast<cm_nn_t>(p_nn);
//            cm_nn->set_learning_rate(getLearningRate());
//            cm_nn->set_momentum(momentum);
//          }
//          break;
//
//        case TrainingMethod::AdaDelta:
//          {
//            boost::shared_ptr<adadelta_nn_t> ad_nn = boost::dynamic_pointer_cast<adadelta_nn_t>(p_nn);
//            ad_nn->set_epsilon(getLearningRate());
//            ad_nn->set_decay_rate(0.95);
//          }
//          break;
//
//        default:
//          dlog(Severity::Warning) << "Training method " << getMethod() << " has not been implemented.";
//        }
//        nn.update_model(weightcost);
      }
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / batchCount;

    if (monitor)
      monitor->reportProgress(100 * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
