/*
 * Train.cpp
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#include "Train.h"

namespace gml {

namespace encoder {


BeginPropertyDefinitions(Train)

  ReflectableBase(DefaultWorkflowElement<Train>)

  WorkflowProperty(InitialModel, Input("ENN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(TrialEpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(FilterBatchSize)
  WorkflowProperty(SubRegionCount, Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(Objective, Enumerator<Type>())
  WorkflowProperty(SensitivityRatio)
  WorkflowProperty(SharedBiasTerms, Flag())

  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(LearningRates, NotEmpty<Type>())
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(WeightCosts)
  WorkflowProperty(InitialWeights, Description("If given, these weights will be tested as initial weights and will override the initial weights."))
  WorkflowProperty(RandomizeTraining, Flag())

  WorkflowProperty(AugmentedChannels)
  WorkflowProperty(ContrastSd)
  WorkflowProperty(BrightnessSd)
  WorkflowProperty(GammaSd)

  WorkflowProperty(BestOfN, Description("If greater than 0, the best and the worst of the last BestOfN models will be selected. This requires the batch size to be equal to the training set size."))

  WorkflowProperty(Model, Output("ENN"))
  WorkflowProperty(Model2, Output("ENN2"))
  WorkflowProperty(BestModel, Output("BENN"))
  WorkflowProperty(WorstModel, Output("WENN"))
  WorkflowProperty(AugmentedSet, Output("AS"))

EndPropertyDefinitions

Train::Train() : _EpochCount(100), _TrialEpochCount(20), _BatchSize(50), _SubRegionCount(tbblas::seq<host_tensor_t::dimCount>(1)),
  _SensitivityRatio(0.5), _SharedBiasTerms(true), _LearningDecay(50), _WeightCosts(0.0002), _RandomizeTraining(true),
  _ContrastSd(0), _BrightnessSd(0), _GammaSd(0), _BestOfN(-1)
{
  setLabel("Train");
  _LearningRates.push_back(0.0001);
}

TrainChecker trainChecker;

} /* namespace encoder */

} /* namespace gml */
