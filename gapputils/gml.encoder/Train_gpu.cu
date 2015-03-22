/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/encoder.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/random.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

namespace gml {

namespace encoder {

TrainChecker::TrainChecker() {
  Train test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(TrialEpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(SubRegionCount, test);
  CHECK_MEMORY_LAYOUT2(Objective, test);
  CHECK_MEMORY_LAYOUT2(SensitivityRatio, test);
  CHECK_MEMORY_LAYOUT2(SharedBiasTerms, test);

  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(LearningRates, test);
  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(WeightCosts, test);
  CHECK_MEMORY_LAYOUT2(InitialWeights, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);

  CHECK_MEMORY_LAYOUT2(AugmentedChannels, test);
  CHECK_MEMORY_LAYOUT2(ContrastSd, test);
  CHECK_MEMORY_LAYOUT2(BrightnessSd, test);
  CHECK_MEMORY_LAYOUT2(GammaSd, test);

  CHECK_MEMORY_LAYOUT2(BestOfN, test);
  CHECK_MEMORY_LAYOUT2(SaveEvery, test);

  CHECK_MEMORY_LAYOUT2(CurrentEpoch, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Model2, test);
  CHECK_MEMORY_LAYOUT2(BestModel, test);
  CHECK_MEMORY_LAYOUT2(WorstModel, test);
  CHECK_MEMORY_LAYOUT2(AugmentedSet, test);
}

void Train::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  if (_BestOfN > 0) {
    if (_BatchSize != getTrainingSet()->size()) {
      dlog(Severity::Warning) << "Batch size must be equal to the training size for best of n  model selection. Aborting!";
      return;
    }

    if (_EpochCount < _BestOfN) {
      dlog(Severity::Warning) << "BestOfN must not be greater than EpochCount. Aborting!";
      return;
    }
  }

  // Prepare data
  v_host_tensor_t& tensors = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;
  size_t epochCount = getEpochCount();

  value_t epsilon, initialWeight = 0, bestEpsilon, bestError, bestWeight = 0;
  size_t current;

  std::vector<double> learningRates = getLearningRates();
  std::vector<double> initialWeights = getInitialWeights();

#ifdef AUGMENTED_SET
  boost::shared_ptr<v_host_tensor_t> augmentedSet(new v_host_tensor_t());
  newState->setAugmentedSet(augmentedSet);
#endif

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
      model->set_shared_bias(getSharedBiasTerms());

//      if (initialWeight > 0) {
//        model_t::nn_layer_t& layer = *model->nn_layers()[model->nn_layers().size() - 1];
//
//        random_tensor2<value_t, 2, false, normal<value_t> > randn(layer.weights().size());
//        tensor<value_t, 2> W = initialWeight * randn();
//        layer.set_weights(W);
//      }

      tbblas::deeplearn::encoder<value_t, dimCount> encoder(*model, _SubRegionCount);
      encoder.set_objective_function(getObjective());
      encoder.set_sensitivity_ratio(getSensitivityRatio());
      for (size_t i = 0; i < model->cnn_encoders().size() + model->dnn_decoders().size() && i < getFilterBatchSize().size(); ++i)
        encoder.set_batch_length(i, getFilterBatchSize()[i]);


      dlog() << "Preparation finished. Starting training.";

      value_t error, learningDecay = 1, PPV, DSC, TPR, TNR, ACC, worstError = 0;
      tensor_t v, target, channel;

      // For data augmentation
      boost::mt19937 rng; // I don't seed it on purpose (it's not relevant)
      boost::normal_distribution<> nd(0.0, 1.0);
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

      for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
        newState->setCurrentEpoch(iEpoch);

        ACC = PPV = DSC = TPR = TNR = error = 0;

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
            if (getRandomizeTraining())
              current = rand() % tensors.size();
            else
              current = iSample + iBatch * batchSize;

            v = *tensors[current];
            target = *labels[current];

            // Perform data augmentation
            dim_t channelSize = v.size();
            channelSize[dimCount - 1] = 1;

            dim_t topleft = seq<dimCount>(0);
            topleft[dimCount - 1] = 1;

            for (int iChannel = 0; iChannel < v.size()[dimCount - 1]; ++iChannel) {

              // Skip channel if no augmentation was requested for the current channel
              if (iChannel >= _AugmentedChannels.size() || !_AugmentedChannels[iChannel])
                continue;

              channel = v[topleft * iChannel, channelSize];

              const value_t slope = exp(var_nor() * _ContrastSd);
              const value_t gamma = exp(var_nor() * _GammaSd);
              const value_t intercept = var_nor() * _BrightnessSd;

              const value_t minValue = min(channel);
              const value_t maxValue = max(channel);

              // Standardize
              channel = (channel - minValue) / (maxValue - minValue);

              // Calculate new contrast
              channel = (slope * pow(channel, gamma) - value_t(0.5)) + value_t(0.5) + intercept;

              // Diversify
              channel = channel * (maxValue - minValue) + minValue;

              v[topleft * iChannel, channelSize] = channel;
            }

#ifdef AUGMENTED_SET
            if (iEpoch == 0 && iLearningRate == learningRates.size()) {
              augmentedSet->push_back(boost::make_shared<host_tensor_t>(v));
              tbblas::synchronize();
            }
#endif

            encoder.inputs() = v;
            encoder.update_gradient(target);


            PPV += sum((target > 0.5) * (encoder.outputs() > 0.5)) / sum(encoder.outputs() > 0.5);
            DSC += 2 * sum ((target > 0.5) * (encoder.outputs() > 0.5)) / (sum(target > 0.5) + sum(encoder.outputs() > 0.5));
            TPR += (sum((target > 0.5) * (encoder.outputs() > 0.5)) + 1e-8) / (sum(target > 0.5) + 1e-8);
            TNR += sum((target < 0.5) * (encoder.outputs() < 0.5)) / sum(target < 0.5);
            ACC += sum((target > 0.5) == (encoder.outputs() > 0.5)) / (value_t)target.count();

            if (getObjective() == tbblas::deeplearn::objective_function::SenSpe) {
              const value_t sen = dot(encoder.outputs() - target, (encoder.outputs() - target) * (target > 0.5)) / sum(target > 0.5);
              const value_t spe = dot(encoder.outputs() - target, (encoder.outputs() - target) * (target < 0.5)) / sum(target < 0.5);
              error += _SensitivityRatio * sen + (1.0 - _SensitivityRatio) * spe;
            } else {
              error += sqrt(dot(encoder.outputs() - target, encoder.outputs() - target) / target.count());
            }

            if (sum(target > 0.5) == 0) {
              dlog(Severity::Warning) << "No lesions detected: " << current;
            }
          }

          if (_BestOfN > 0 && iLearningRate == learningRates.size() && iEpoch + _BestOfN >= epochCount) {

            if (iEpoch + _BestOfN == epochCount) {
              dlog(Severity::Trace) << "Initialize best and worst model.";
              encoder.write_model_to_host();
              bestError = worstError = error;
              newState->setBestModel(boost::make_shared<model_t>(*model));
              newState->setWorstModel(boost::make_shared<model_t>(*model));
            } else if (error < bestError) {
              dlog(Severity::Trace) << "Found better model.";
              bestError = error;
              encoder.write_model_to_host();
              newState->setBestModel(boost::make_shared<model_t>(*model));
            } else if (error > worstError) {
              dlog(Severity::Trace) << "Found worse model";
              worstError = error;
              encoder.write_model_to_host();
              newState->setWorstModel(boost::make_shared<model_t>(*model));
            }
          }

          switch (getMethod()) {
          case TrainingMethod::Momentum:
            encoder.momentum_step(epsilon * learningDecay, epsilon * learningDecay, momentum, weightcost);
            break;

          case TrainingMethod::AdaDelta:
            encoder.adadelta_step(epsilon * learningDecay, epsilon * learningDecay, momentum, weightcost);
            break;

          case TrainingMethod::Adam:
            encoder.adam_step(epsilon * learningDecay, 0.1, 0.001, 1e-8, 1, weightcost);
            break;

          case TrainingMethod::AdamDecay:
            encoder.adam_step(epsilon * learningDecay, 0.1, 0.001, 1e-8, 1e-8, weightcost);
            break;
          }
        }

        if (_SaveEvery > 0 && iLearningRate == learningRates.size() && iEpoch % _SaveEvery == 0) {
          dlog(Severity::Trace) << "Saving model at epoch " << iEpoch;
          encoder.write_model_to_host();
          newState->setModel(boost::make_shared<model_t>(*model));
        }

        dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << epochCount << " epochs: " << error / tensors.size()
            << " (PPV = " << PPV / tensors.size() << ", DSC = " << DSC / tensors.size() << ", TPR = " << TPR / tensors.size() << ", TNR = " << TNR / tensors.size() << ", ACC = " << ACC / tensors.size() << ")";

        if (monitor) {
          const bool parameterTuning = (initialWeights.size() || learningRates.size() > 1);
          const int totalEpochs = getTrialEpochCount() * max(1, (int)initialWeights.size()) * learningRates.size() * parameterTuning + getEpochCount();
          const int currentEpoch = iEpoch + (iLearningRate + iWeight * learningRates.size()) * getTrialEpochCount() * parameterTuning;
          monitor->reportProgress(100.0 * (currentEpoch + 1) / totalEpochs, (_SaveEvery > 0) && (iEpoch % _SaveEvery == 0));
        }

        if (iLearningRate == learningRates.size() && iEpoch + 2 == epochCount) {
          dlog(Severity::Trace) << "Last epoch detected. Saving pre-model.";
          encoder.write_model_to_host();
          newState->setModel2(boost::make_shared<model_t>(*model));
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
