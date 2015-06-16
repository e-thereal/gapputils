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
#include <tbblas/deeplearn/opt/rms_prop.hpp>

#include <tbblas/deeplearn/opt/trainer_base.hpp>
#include <tbblas/deeplearn/opt/first_order.hpp>
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
//  CHECK_MEMORY_LAYOUT2(TrialEpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(SubRegionCount, test);
  CHECK_MEMORY_LAYOUT2(Objective, test);
  CHECK_MEMORY_LAYOUT2(SensitivityRatio, test);
  CHECK_MEMORY_LAYOUT2(SharedBiasTerms, test);

  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(Parameters, test);
//  CHECK_MEMORY_LAYOUT2(LearningRates, test);
//  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(WeightCosts, test);
//  CHECK_MEMORY_LAYOUT2(InitialWeights, test);
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

  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;
  size_t epochCount = getEpochCount();

  size_t current;

#ifdef AUGMENTED_SET
  boost::shared_ptr<v_host_tensor_t> augmentedSet(new v_host_tensor_t());
  newState->setAugmentedSet(augmentedSet);
#endif

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));
  model->set_shared_bias(getSharedBiasTerms());

  td::encoder<value_t, dimCount> encoder(*model, _SubRegionCount);
  encoder.set_objective_function(getObjective());
  encoder.set_sensitivity_ratio(getSensitivityRatio());
  for (size_t i = 0; i < model->cnn_encoders().size() + model->dnn_decoders().size() && i < getFilterBatchSize().size(); ++i)
    encoder.set_batch_length(i, getFilterBatchSize()[i]);

  typedef td::opt::trainer_base<value_t, dimCount> trainer_base_t;
  typedef td::opt::first_order<value_t, dimCount, td::opt::classic_momentum<value_t> > cm_trainer_t;
  typedef td::opt::first_order<value_t, dimCount, td::opt::nesterov_momentum<value_t> > nag_trainer_t;
  typedef td::opt::first_order<value_t, dimCount, td::opt::adagrad<value_t> > adagrad_trainer_t;
  typedef td::opt::first_order<value_t, dimCount, td::opt::adadelta<value_t> > adadelta_trainer_t;
  typedef td::opt::first_order<value_t, dimCount, td::opt::adam<value_t> > adam_trainer_t;
  typedef td::opt::first_order<value_t, dimCount, td::opt::rms_prop<value_t> > rp_trainer_t;
  typedef td::opt::hessian_free<value_t, dimCount> hf_trainer_t;

  boost::shared_ptr<trainer_base_t> base_trainer;

  switch (getMethod()) {
  case TrainingMethod::ClassicMomentum:
    base_trainer = boost::make_shared<cm_trainer_t>(boost::ref(encoder));
    break;

  case TrainingMethod::NesterovMomentum:
    base_trainer = boost::make_shared<nag_trainer_t>(boost::ref(encoder));
    break;

  case TrainingMethod::AdaGrad:
    base_trainer = boost::make_shared<adagrad_trainer_t>(boost::ref(encoder));
    break;

  case TrainingMethod::AdaDelta:
    {
      AdaDeltaParameters* parameters = dynamic_cast<AdaDeltaParameters*>(_Parameters.get());
      boost::shared_ptr<adadelta_trainer_t> trainer(new adadelta_trainer_t(encoder));
      trainer->set_decay_rate(parameters->getDecayRate());
      trainer->set_epsilon(parameters->getEpsilon());
      base_trainer = trainer;
    }
    break;

  case TrainingMethod::Adam:
    {
      AdamParameters* parameters = dynamic_cast<AdamParameters*>(_Parameters.get());
      boost::shared_ptr<adam_trainer_t> trainer(new adam_trainer_t(encoder));
      trainer->set_alpha(parameters->getAlpha());
      trainer->set_beta1(parameters->getBeta1());
      trainer->set_beta2(parameters->getBeta2());
      trainer->set_epsilon(parameters->getEpsilon());
      base_trainer = trainer;
    }
    break;

  case TrainingMethod::RmsProp:
    base_trainer = boost::make_shared<rp_trainer_t>(boost::ref(encoder));
    break;

  case TrainingMethod::HessianFree:
    {
      HessianFreeParameters* parameters = dynamic_cast<HessianFreeParameters*>(_Parameters.get());
      boost::shared_ptr<hf_trainer_t> trainer(new hf_trainer_t(encoder));
      trainer->set_iteration_count(parameters->getIterationCount());
      trainer->set_lambda(parameters->getInitialLambda());
      trainer->set_zeta(parameters->getZeta());
      base_trainer = trainer;
    }
    break;

  default:
    dlog(Severity::Warning) << "Unsupported trainer selected. Aborting!";
    return;
  }

  base_trainer->set_weightcost(getWeightCosts());

  v_host_tensor_t inputBatch(batchSize), targetBatch(batchSize);

  dlog() << "Preparation finished. Starting training.";

  value_t error, PPV, DSC, TPR, TNR, ACC;
  tensor_t v, target, channel;

  // For data augmentation
  boost::mt19937 rng; // I don't seed it on purpose (it's not relevant)
  boost::normal_distribution<> nd(0.0, 1.0);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

  for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
    newState->setCurrentEpoch(iEpoch);

    // Update epoch dependent parameters
    switch (getMethod()) {
    case TrainingMethod::ClassicMomentum:
      {
        boost::shared_ptr<cm_trainer_t> trainer = boost::dynamic_pointer_cast<cm_trainer_t>(base_trainer);
        MomentumParameters* parameters = dynamic_cast<MomentumParameters*>(_Parameters.get());
        trainer->set_learning_rate(parameters->getLearningRate(iEpoch));
        trainer->set_momentum(parameters->getMomentum(iEpoch));
      }
      break;

    case TrainingMethod::NesterovMomentum:
      {
        boost::shared_ptr<nag_trainer_t> trainer = boost::dynamic_pointer_cast<nag_trainer_t>(base_trainer);
        MomentumParameters* parameters = dynamic_cast<MomentumParameters*>(_Parameters.get());
        trainer->set_learning_rate(parameters->getLearningRate(iEpoch));
        trainer->set_momentum(parameters->getMomentum(iEpoch));
      }
      break;

    case TrainingMethod::AdaGrad:
      {
        boost::shared_ptr<adagrad_trainer_t> trainer = boost::dynamic_pointer_cast<adagrad_trainer_t>(base_trainer);
        AdaGradParameters* parameters = dynamic_cast<AdaGradParameters*>(_Parameters.get());
        trainer->set_learning_rate(parameters->getLearningRate(iEpoch));
        trainer->set_epsilon(parameters->getEpsilon());
      }
      break;

    case TrainingMethod::RmsProp:
      {
        boost::shared_ptr<rp_trainer_t> trainer = boost::dynamic_pointer_cast<rp_trainer_t>(base_trainer);
        MomentumParameters* parameters = dynamic_cast<MomentumParameters*>(_Parameters.get());
        trainer->set_learning_rate(parameters->getLearningRate(iEpoch));
        trainer->set_decay_rate(parameters->getMomentum(iEpoch));
      }
      break;
    }

    ACC = PPV = DSC = TPR = TNR = error = 0;

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

#define MONITOR_TRAINING

#ifdef MONITOR_TRAINING
        encoder.inputs() = v;
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
#endif
      }

//      trainer.check_loss(inputBatch, targetBatch);
//      trainer.check_gradient(inputBatch, targetBatch, epsilon);
//      trainer.check_Gv(inputBatch, targetBatch, epsilon, _RandomizeTraining);
//      return;

    }

    base_trainer->update(inputBatch, targetBatch);

    if (_SaveEvery > 0 && iEpoch % _SaveEvery == 0) {
      dlog(Severity::Trace) << "Saving model at epoch " << iEpoch;
      encoder.write_model_to_host();
      newState->setModel(boost::make_shared<model_t>(*model));
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << epochCount << " epochs: " << error / tensors.size()
        << " (PPV = " << PPV / tensors.size() << ", DSC = " << DSC / tensors.size() << ", TPR = " << TPR / tensors.size() << ", TNR = " << TNR / tensors.size() << ", ACC = " << ACC / tensors.size() << ")";

    if (monitor) {
      monitor->reportProgress(100.0 * (iEpoch + 1) / epochCount, (_SaveEvery > 0) && (iEpoch % _SaveEvery == 0));
    }
  }

  newState->setModel(model);
}

}

}
