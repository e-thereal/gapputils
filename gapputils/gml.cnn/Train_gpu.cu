/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/cnn.hpp>
#include <tbblas/deeplearn/opt/classic_momentum.hpp>
#include <tbblas/deeplearn/opt/adadelta.hpp>

#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/random.hpp>

namespace gml {

namespace cnn {

TrainChecker::TrainChecker() {
  Train test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(TrialEpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);

  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(LearningRates, test);
  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(WeightCosts, test);
  CHECK_MEMORY_LAYOUT2(InitialWeights, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
}

namespace td = tbblas::deeplearn;

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

  // Prepare data
  v_host_tensor_t& tensors = *getTrainingSet();
  v_data_t& labels = *getLabels();

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;
  size_t epochCount = getEpochCount();

  value_t epsilon, initialWeight = 0, bestEpsilon, bestError, bestWeight = 0;

  std::vector<double> learningRates = getLearningRates();
  std::vector<double> initialWeights = getInitialWeights();

  for (int iWeight = 0; iWeight < initialWeights.size() || (iWeight == 0 && initialWeights.size() == 0); ++iWeight) {
    for (size_t iLearningRate = 0; iLearningRate < learningRates.size() + 1; ++iLearningRate) {

      // iLearningRate == learningRate.size() marks the final run, this is only done when all the weights were tried
      if (iLearningRate == learningRates.size() && iWeight < (int)initialWeights.size() - 1) {
        continue;
      }

      // if only one weight and one learning rate is given, do the final run immediately
      if (iWeight == 0 && iLearningRate == 0 && initialWeights.size() <= 1 && learningRates.size() == 1) {
        bestEpsilon = learningRates[0];
        if (initialWeights.size())
          bestWeight = initialWeights[0];
        continue;
      }

      if (iLearningRate < learningRates.size()) {
        epsilon = learningRates[iLearningRate];
        if (initialWeights.size())
          initialWeight = initialWeights[iWeight];
        epochCount = getTrialEpochCount();
        dlog(Severity::Message) << "Trying learning rate of " << learningRates[iLearningRate] << " and initial weight of " << initialWeight;
      } else {
        epsilon = bestEpsilon;
        if (initialWeights.size())
          initialWeight = bestWeight;
        dlog(Severity::Message) << "Final run with learning rate: " << bestEpsilon << " and initial weight of " << initialWeight;
        epochCount = getEpochCount();
      }

      boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

      if (initialWeight > 0) {
        model_t::nn_layer_t& layer = *model->nn_layers()[model->nn_layers().size() - 1];

        random_tensor2<value_t, 2, false, normal<value_t> > randn(layer.weights().size());
        tensor<value_t, 2> W = initialWeight * randn();
        layer.set_weights(W);
      }

      typedef td::cnn_base<value_t, dimCount> cnn_base_t;
      typedef td::cnn<value_t, dimCount, td::opt::classic_momentum<value_t> > cm_cnn_t;
      typedef td::cnn<value_t, dimCount, td::opt::adadelta<value_t> > adadelta_cnn_t;

      boost::shared_ptr<cnn_base_t> p_cnn;

      switch (getMethod()) {
      case TrainingMethod::ClassicMomentum:
        p_cnn = boost::make_shared<cm_cnn_t>(boost::ref(*model));
        break;

      case TrainingMethod::AdaDelta:
        p_cnn = boost::make_shared<adadelta_cnn_t>(boost::ref(*model));
        break;

      default:
        dlog(Severity::Warning) << "Unsupported optimization method. Aborting!";
        return;
      }

      cnn_base_t& cnn = *p_cnn;
      for (size_t i = 0; i < model->cnn_layers().size() && i < getFilterBatchSize().size(); ++i)
        cnn.set_batch_length(i, getFilterBatchSize()[i]);

      dlog() << "Preparation finished. Starting training.";

      value_t error, learningDecay = 1;
      matrix_t target(1, model->hiddens_count());
      tensor_t v;

      for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

        error = 0;

        // Learning decay only during final learning
        if (getLearningDecay() > 1 && iLearningRate == learningRates.size()) {
          learningDecay = (value_t)getLearningDecay() / ((value_t)getLearningDecay() + (value_t)iEpoch);
        }

        if (iEpoch < 10)
          momentum = initialmomentum;
        else
          momentum = finalmomentum;

        for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

          for (int iSample = 0; iSample < batchSize; ++iSample) {
            const int current = iSample + iBatch * batchSize;
            thrust::copy(labels[current]->begin(), labels[current]->end(), target.begin());
            v = *tensors[current];
            cnn.visibles() = v;
            cnn.normalize_visibles();
            cnn.update_gradient(target);
            error += sqrt(dot(cnn.hiddens() - target, cnn.hiddens() - target));
          }

          switch (getMethod()) {
          case TrainingMethod::ClassicMomentum:
            {
              boost::shared_ptr<cm_cnn_t> cm_cnn = boost::dynamic_pointer_cast<cm_cnn_t>(p_cnn);
              cm_cnn->set_learning_rate(epsilon * learningDecay);
              cm_cnn->set_momentum(momentum);
              cm_cnn->update_model(weightcost);
            }
            break;

          case TrainingMethod::AdaDelta:
            {
              boost::shared_ptr<adadelta_cnn_t> adadelta_cnn = boost::dynamic_pointer_cast<adadelta_cnn_t>(p_cnn);
              adadelta_cnn->set_epsilon(epsilon * learningDecay);
              adadelta_cnn->set_decay_rate(momentum);
              adadelta_cnn->update_model(weightcost);
            }
            break;
          }
        }

        dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << epochCount << " epochs: " << error / tensors.size();

        if (monitor) {
          const int totalEpochs = getTrialEpochCount() * max(1, (int)initialWeights.size()) * learningRates.size() + getEpochCount();
          const int currentEpoch = iEpoch + (iLearningRate + iWeight * learningRates.size()) * getTrialEpochCount();
          monitor->reportProgress(100 * (currentEpoch + 1) / totalEpochs);
        }
      }

      if (iLearningRate < learningRates.size()) {
        if (iLearningRate == 0 && iWeight == 0 || !(error > bestError)) {   // using not greater instead of lesser to handle nan case.
          bestError = error;
          bestEpsilon = epsilon;
          bestWeight = initialWeight;
          dlog(Severity::Message) << "Found better learning rate: " << epsilon << " with an error of " << bestError / tensors.size() << ".";
        }
      } else {
        newState->setModel(model);
      }
    }
  }
}

}

}
