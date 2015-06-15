/*
 * Train.cpp
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#include "Train.h"

#include <capputils/EventHandler.h>
#include <capputils/attributes/DummyAttribute.h>
#include <gapputils/attributes/GroupAttribute.h>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(OptimizationParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(MomentumParameters)
  DefineProperty(LearningRate)
  DefineProperty(LearningDecayEpochs)
  DefineProperty(InitialMomentum)
  DefineProperty(FinalMomentum)
  DefineProperty(MomentumDecayEpochs)
EndPropertyDefinitions

MomentumParameters::MomentumParameters() : _LearningRate(0.0001), _LearningDecayEpochs(-1), _InitialMomentum(0.5), _FinalMomentum(0.9), _MomentumDecayEpochs(20) { }

float MomentumParameters::getLearningRate(int epoch) const {
  if (_LearningDecayEpochs > 0)
    return _LearningRate * (float)_LearningDecayEpochs / ((float)_LearningDecayEpochs + (float)epoch);
  else
    return _LearningRate;
}

float MomentumParameters::getMomentum(int epoch) const {
  if (epoch < _MomentumDecayEpochs) {
    const float t = (float)epoch / (float)_MomentumDecayEpochs;
    return (1.0 - t) * _InitialMomentum + t * _FinalMomentum;
  } else {
    return _FinalMomentum;
  }
}

BeginPropertyDefinitions(AdaGradParameters)
  DefineProperty(LearningRate)
  DefineProperty(LearningDecayEpochs)
  DefineProperty(Epsilon)
EndPropertyDefinitions

AdaGradParameters::AdaGradParameters() : _LearningRate(0.0001), _LearningDecayEpochs(-1), _Epsilon(1e-8) { }

float AdaGradParameters::getLearningRate(int epoch) const {
  if (_LearningDecayEpochs > 0)
    return _LearningRate * (float)_LearningDecayEpochs / ((float)_LearningDecayEpochs + (float)epoch);
  else
    return _LearningRate;
}

BeginPropertyDefinitions(AdaDeltaParameters)
  DefineProperty(Epsilon)
  DefineProperty(DecayRate)
EndPropertyDefinitions

AdaDeltaParameters::AdaDeltaParameters() : _Epsilon(1e-8), _DecayRate(0.9) { }

BeginPropertyDefinitions(AdamParameters)
  DefineProperty(Alpha)
  DefineProperty(Beta1)
  DefineProperty(Beta2)
  DefineProperty(Epsilon)
EndPropertyDefinitions

AdamParameters::AdamParameters() : _Alpha(0.0002), _Beta1(0.1), _Beta2(0.001), _Epsilon(1e-8) { }

BeginPropertyDefinitions(HessianFreeParameters)
  DefineProperty(ConjugateGradientIterations)
  DefineProperty(InitialLambda)
  DefineProperty(Zeta)
EndPropertyDefinitions

HessianFreeParameters::HessianFreeParameters() : _ConjugateGradientIterations(20), _InitialLambda(1), _Zeta(0.9) { }

int Train::methodId;

BeginPropertyDefinitions(Train)

  ReflectableBase(DefaultWorkflowElement<Train>)

  WorkflowProperty(EpochCount, Group("Optimization"))
  WorkflowProperty(TrialEpochCount, Group("Optimization"))
  WorkflowProperty(BatchSize, Group("Optimization"))
  WorkflowProperty(FilterBatchSize, Group("Performance"))
  WorkflowProperty(SubRegionCount, Group("Performance"), Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(Objective, Enumerator<Type>())
  WorkflowProperty(SensitivityRatio)
  WorkflowProperty(SharedBiasTerms, Flag())

  WorkflowProperty(Method, Enumerator<Type>(), Group("Optimization"), Dummy(methodId = Id))
  WorkflowProperty(Parameters, Reflectable<Type>(), Group("Optimization"))
  WorkflowProperty(LearningRates, NotEmpty<Type>(), Group("Optimization"))
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."), Group("Optimization"))
  WorkflowProperty(InitialMomentum, Description("Momentum used for the first epoch."), Group("Optimization"))
  WorkflowProperty(FinalMomentum, Group("Optimization"))
  WorkflowProperty(MomentumDecayEpochs, Group("Optimization"))
  WorkflowProperty(WeightCosts, Group("Optimization"))
  WorkflowProperty(InitialWeights, Description("If given, these weights will be tested as initial weights and will override the initial weights."), Group("Optimization"))
  WorkflowProperty(RandomizeTraining, Flag(), Group("Optimization"))

  WorkflowProperty(AugmentedChannels, Group("Data augmentation"))
  WorkflowProperty(ContrastSd, Group("Data augmentation"))
  WorkflowProperty(BrightnessSd, Group("Data augmentation"))
  WorkflowProperty(GammaSd, Group("Data augmentation"))

  WorkflowProperty(SaveEvery, Description("If greater than 0, the model is saved every SaveEvery epochs."))

  WorkflowProperty(InitialModel, Input("ENN"), NotNull<Type>(), Group("Input/output"))
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>(), Group("Input/output"))
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>(), Group("Input/output"))
  WorkflowProperty(Model, Output("ENN"), Group("Input/output"))
  WorkflowProperty(AugmentedSet, Output("AS"), Group("Input/output"))
  WorkflowProperty(CurrentEpoch, NoParameter(), Group("Input/output"))

EndPropertyDefinitions

Train::Train() : _EpochCount(100), _TrialEpochCount(20), _BatchSize(50), _SubRegionCount(tbblas::seq<host_tensor_t::dimCount>(1)),
  _SensitivityRatio(0.5), _SharedBiasTerms(true), _Parameters(new MomentumParameters()), _LearningDecay(50),
  _InitialMomentum(0.5), _FinalMomentum(0.9), _MomentumDecayEpochs(20),
  _WeightCosts(0.0002), _RandomizeTraining(true),
  _ContrastSd(0), _BrightnessSd(0), _GammaSd(0), _SaveEvery(-1), _CurrentEpoch(0)
{
  setLabel("Train");
  _LearningRates.push_back(0.0001);

  Changed.connect(EventHandler<Train>(this, &Train::changedHandler));
}

void Train::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == methodId) {
    switch (getMethod()) {
    case TrainingMethod::ClassicMomentum:
    case TrainingMethod::NesterovMomentum:
    case TrainingMethod::RmsProp:
      if (!boost::dynamic_pointer_cast<MomentumParameters>(getParameters()))
        setParameters(boost::make_shared<MomentumParameters>());
      break;

    case TrainingMethod::AdaGrad:
      if (!boost::dynamic_pointer_cast<AdaGradParameters>(getParameters()))
        setParameters(boost::make_shared<AdaGradParameters>());
      break;

    case TrainingMethod::AdaDelta:
      if (!boost::dynamic_pointer_cast<AdaDeltaParameters>(getParameters()))
        setParameters(boost::make_shared<AdaDeltaParameters>());
      break;

    case TrainingMethod::Adam:
      if (!boost::dynamic_pointer_cast<AdamParameters>(getParameters()))
        setParameters(boost::make_shared<AdamParameters>());
      break;

    case TrainingMethod::HessianFree:
      if (!boost::dynamic_pointer_cast<HessianFreeParameters>(getParameters()))
        setParameters(boost::make_shared<HessianFreeParameters>());
      break;

    default:
      if (!boost::dynamic_pointer_cast<OptimizationParameters>(getParameters()))
        setParameters(boost::make_shared<OptimizationParameters>());
      break;
    }
  }
}

TrainChecker trainChecker;

} /* namespace encoder */

} /* namespace gml */
