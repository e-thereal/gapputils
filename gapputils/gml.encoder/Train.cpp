/*
 * Train.cpp
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#include "Train.h"

#include <gapputils/attributes/GroupAttribute.h>

namespace gml {

namespace encoder {


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

  WorkflowProperty(Method, Enumerator<Type>(), Group("Optimization"))
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
  _SensitivityRatio(0.5), _SharedBiasTerms(true), _LearningDecay(50),
  _InitialMomentum(0.5), _FinalMomentum(0.9), _MomentumDecayEpochs(20),
  _WeightCosts(0.0002), _RandomizeTraining(true),
  _ContrastSd(0), _BrightnessSd(0), _GammaSd(0), _SaveEvery(-1), _CurrentEpoch(0)
{
  setLabel("Train");
  _LearningRates.push_back(0.0001);
}

TrainChecker trainChecker;

} /* namespace encoder */

} /* namespace gml */
