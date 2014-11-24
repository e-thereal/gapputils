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
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRate)
  WorkflowProperty(BiasLearningRate)
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(WeightDecay)
  WorkflowProperty(InitialWeights)
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
  WorkflowProperty(FindLearningRate, Flag())
  WorkflowProperty(TrialLearningRates)
  WorkflowProperty(TrialEpochCount)

  WorkflowProperty(Model, Output("RBM"))

EndPropertyDefinitions

Trainer::Trainer()
 : _AutoCreateMask(false), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01), _BiasLearningRate(0.03), _LearningDecay(-1), _WeightDecay(0.0002), _InitialWeights(0.01), _InitialVisible(0.0), _InitialHidden(0.0), _HiddenDropout(0),
   _SparsityTarget(0.1), _SparsityWeight(0.1), _NormalizeIndividualUnits(true), _ShuffleTrainingSet(true), _ShowWeights(0), _ShowEvery(1), _FindLearningRate(false), _TrialEpochCount(10)
{
  setLabel("Trainer");
}

TrainerChecker trainerChecker;

}

}
