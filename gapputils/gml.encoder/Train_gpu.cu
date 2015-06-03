/*
 * Train_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

#include <tbblas/deeplearn/encoder.hpp>
#include <tbblas/deeplearn/opt/classic_momentum.hpp>
#include <tbblas/deeplearn/opt/nesterov_momentum.hpp>
#include <tbblas/deeplearn/opt/adadelta.hpp>
#include <tbblas/deeplearn/opt/adagrad.hpp>
#include <tbblas/deeplearn/opt/adam.hpp>
#include <tbblas/deeplearn/opt/adam2.hpp>
#include <tbblas/deeplearn/opt/rms_prop.hpp>
#include <tbblas/deeplearn/opt/vsgd_fd.hpp>
#include <tbblas/deeplearn/opt/vsgd_fd_v2.hpp>

#include <tbblas/deeplearn/opt/hessian_free.hpp>

#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/random.hpp>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <boost/ref.hpp>

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

  CHECK_MEMORY_LAYOUT2(SaveEvery, test);

  CHECK_MEMORY_LAYOUT2(CurrentEpoch, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(AugmentedSet, test);
}

namespace td = tbblas::deeplearn;

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

  // Prepare data
  v_host_tensor_t& tensors = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
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

      typedef td::encoder_base<value_t, dimCount> encoder_base_t;
      typedef td::encoder<value_t, dimCount, td::opt::classic_momentum<value_t> > cm_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::nesterov_momentum<value_t> > nag_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::adagrad<value_t> > adagrad_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::adadelta<value_t> > adadelta_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::adam<value_t> > adam_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::adam2<value_t> > adam2_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::rms_prop<value_t> > rp_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::vsgd_fd<value_t> > vsgdfd_encoder_t;
      typedef td::encoder<value_t, dimCount, td::opt::vsgd_fd_v2<value_t> > vsgdfd2_encoder_t;
      typedef td::encoder<value_t, dimCount> default_encoder_t;

//      boost::shared_ptr<encoder_base_t> p_encoder;
//
//      switch (getMethod()) {
//      case TrainingMethod::ClassicMomentum:
//        p_encoder = boost::make_shared<cm_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::NesterovMomentum:
//        p_encoder = boost::make_shared<nag_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::AdaGrad:
//        p_encoder = boost::make_shared<adagrad_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::AdaDelta:
//        p_encoder = boost::make_shared<adadelta_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::Adam:
//        p_encoder = boost::make_shared<adam_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::AdamDecay:
//        p_encoder = boost::make_shared<adam2_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::RmsProp:
//        p_encoder = boost::make_shared<rp_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::vSGD_fd:
//        p_encoder = boost::make_shared<vsgdfd_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      case TrainingMethod::vSGD_fd_v2:
//        p_encoder = boost::make_shared<vsgdfd2_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//        break;
//
//      default:
//        p_encoder = boost::make_shared<default_encoder_t>(boost::ref(*model), boost::ref(_SubRegionCount));
//      }
//
//      encoder_base_t& encoder = *p_encoder;
      td::encoder<value_t, dimCount> encoder(*model, _SubRegionCount);
      encoder.set_objective_function(getObjective());
      encoder.set_sensitivity_ratio(getSensitivityRatio());
      for (size_t i = 0; i < model->cnn_encoders().size() + model->dnn_decoders().size() && i < getFilterBatchSize().size(); ++i)
        encoder.set_batch_length(i, getFilterBatchSize()[i]);

      td::opt::hessian_free<value_t, dimCount> trainer(encoder);
      v_host_tensor_t inputBatch(batchSize), targetBatch(batchSize);

      dlog() << "Preparation finished. Starting training.";

      value_t error, learningDecay = 1, PPV, DSC, TPR, TNR, ACC;
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

        if (iEpoch < getMomentumDecayEpochs()) {
          const value_t t = (value_t)iEpoch / (value_t)getMomentumDecayEpochs();
          momentum = (1 - t) * initialmomentum + t * finalmomentum;
        } else {
          momentum = finalmomentum;
        }

        for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

          for (int iSample = 0; iSample < batchSize; ++iSample) {
            if (getRandomizeTraining())
              current = rand() % tensors.size();
            else
              current = iSample + iBatch * batchSize;

            v = *tensors[current];
            target = *labels[current];

            inputBatch[iSample] = tensors[current];
            targetBatch[iSample] = labels[current];

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
//            encoder.update_gradient(target);
            encoder.infer_outputs();

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

//          trainer.set_alpha(epsilon * learningDecay);
          trainer.set_weightcost(weightcost);
          trainer.update(inputBatch, targetBatch);

//          switch (getMethod()) {
//          case TrainingMethod::ClassicMomentum:
//            {
//              boost::shared_ptr<cm_encoder_t> encoder = boost::dynamic_pointer_cast<cm_encoder_t>(p_encoder);
//              encoder->set_learning_rate(epsilon * learningDecay);
//              encoder->set_momentum(momentum);
//            }
//            break;
//
//          case TrainingMethod::NesterovMomentum:
//            {
//              boost::shared_ptr<nag_encoder_t> encoder = boost::dynamic_pointer_cast<nag_encoder_t>(p_encoder);
//              encoder->set_learning_rate(epsilon * learningDecay);
//              encoder->set_momentum(momentum);
//            }
//            break;
//
//          case TrainingMethod::AdaGrad:
//            {
//              boost::shared_ptr<adagrad_encoder_t> encoder = boost::dynamic_pointer_cast<adagrad_encoder_t>(p_encoder);
//              encoder->set_learning_rate(epsilon * learningDecay);
//              encoder->set_epsilon(momentum);
//            }
//            break;
//
//          case TrainingMethod::AdaDelta:
//            {
//              boost::shared_ptr<adadelta_encoder_t> encoder = boost::dynamic_pointer_cast<adadelta_encoder_t>(p_encoder);
//              encoder->set_epsilon(epsilon * learningDecay);
//              encoder->set_decay_rate(momentum);
//            }
//            break;
//
//          case TrainingMethod::Adam:
//            {
//              boost::shared_ptr<adam_encoder_t> encoder = boost::dynamic_pointer_cast<adam_encoder_t>(p_encoder);
//              encoder->set_alpha(epsilon * learningDecay);
//            }
//            break;
//
//          case TrainingMethod::AdamDecay:
//            {
//              boost::shared_ptr<adam2_encoder_t> encoder = boost::dynamic_pointer_cast<adam2_encoder_t>(p_encoder);
//              encoder->set_alpha(epsilon * learningDecay);
//            }
//            break;
//
//          case TrainingMethod::RmsProp:
//            {
//              boost::shared_ptr<rp_encoder_t> encoder = boost::dynamic_pointer_cast<rp_encoder_t>(p_encoder);
//              encoder->set_learning_rate(epsilon * learningDecay);
//              encoder->set_decay_rate(momentum);
//            }
//            break;
//
//          case TrainingMethod::vSGD_fd:
//            {
//              boost::shared_ptr<vsgdfd_encoder_t> encoder = boost::dynamic_pointer_cast<vsgdfd_encoder_t>(p_encoder);
//              encoder->set_epsilon(epsilon * learningDecay);
//              encoder->set_c(momentum);
//            }
//            break;
//
//          case TrainingMethod::vSGD_fd_v2:
//            {
//              boost::shared_ptr<vsgdfd2_encoder_t> encoder = boost::dynamic_pointer_cast<vsgdfd2_encoder_t>(p_encoder);
//              encoder->set_epsilon(epsilon * learningDecay);
//              encoder->set_c(momentum);
//            }
//            break;
//          }
//          encoder.update_model(weightcost);
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
