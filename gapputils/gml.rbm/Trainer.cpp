/*
 * Trainer.cpp
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#include "Trainer.h"

namespace gml {

namespace rbm {

BeginPropertyDefinitions(Trainer)

  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(TrainingSet, Input("Data"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(DbmLayer, Enumerator<Type>())
  WorkflowProperty(Mask, Input("Mask"))
  WorkflowProperty(AutoCreateMask, Flag(), Description("If checked and no mask is given, then visible units that are always 0 are masked out."))
  WorkflowProperty(HiddenCount)
  WorkflowProperty(SampleHiddens, Flag())
  WorkflowProperty(EpochCount)
  WorkflowProperty(TrialEpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(Method)
  WorkflowProperty(LearningRates, NotEmpty<Type>())
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(WeightDecay)
  WorkflowProperty(InitialWeights, NotEmpty<Type>())
  WorkflowProperty(InitialVisible)
  WorkflowProperty(InitialHidden)
  WorkflowProperty(HiddenDropout)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight)
  WorkflowProperty(VisibleUnitType, Enumerator<Type>())
  WorkflowProperty(HiddenUnitType, Enumerator<Type>())
  WorkflowProperty(NormalizeIndividualUnits, Flag(), Description("If checked, the mean and standard deviation is calculated per unit and not for all units during the normalization of Gaussian visible units."))
  WorkflowProperty(ShuffleTrainingSet, Flag())
  WorkflowProperty(ShowWeights, Description("Only the first ShowWeights features are shown."))
  WorkflowProperty(ShowEvery, Description("Debug output is shown only every ShowEvery epochs."))

  WorkflowProperty(Model, Output("RBM"))

EndPropertyDefinitions

Trainer::Trainer()
 : _AutoCreateMask(false), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(100), _TrialEpochCount(10), _BatchSize(10), _LearningDecay(-1), _WeightDecay(0.0002), _InitialVisible(0.0), _InitialHidden(0.0), _HiddenDropout(0),
   _SparsityTarget(0.1), _SparsityWeight(0.1), _NormalizeIndividualUnits(true), _ShuffleTrainingSet(true), _ShowWeights(0), _ShowEvery(1)
{
  setLabel("Trainer");

  _LearningRates.push_back(0.01);
  _InitialWeights.push_back(0.01);
}

TrainerChecker trainerChecker;

}

}
