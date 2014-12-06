/*
 * TrainPatch.cpp
 *
 *  Created on: Dec 01, 2014
 *      Author: tombr
 */

#include "TrainPatch.h"

namespace gml {

namespace cnn {


BeginPropertyDefinitions(TrainPatch)

  ReflectableBase(DefaultWorkflowElement<TrainPatch>)

  WorkflowProperty(InitialModel, Input("CNN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mask, Input("M"))
  WorkflowProperty(EpochCount)
  WorkflowProperty(TrialEpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(FilterBatchSize)
  WorkflowProperty(PatchCounts, Description("Number of patches in x-, y-, and z-direction used for the multi-patch training."))
  WorkflowProperty(MultiPatchCount, Description("Number of multi-patches per training image."))
  WorkflowProperty(PositiveRatio, Description("Expected probability of drawing a positive sample."))

  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(LearningRates, NotEmpty<Type>())
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(WeightCosts)
  WorkflowProperty(InitialWeights, Description("If given, these weights will be tested as initial weights and will override the initial weights."))
  WorkflowProperty(RandomizeTraining, Flag())
  WorkflowProperty(Model, Output("CNN"))
  WorkflowProperty(Patches, Output("IPs"))
  WorkflowProperty(Targets, Output("TPs"))
  WorkflowProperty(Predictions, Output("PPs"))

EndPropertyDefinitions

TrainPatch::TrainPatch() : _EpochCount(100), _TrialEpochCount(20), _BatchSize(50), _MultiPatchCount(100), _PositiveRatio(0.5), _LearningDecay(50), _WeightCosts(0.0002), _RandomizeTraining(true)
{
  setLabel("Train");
  _LearningRates.push_back(0.0001);
}

TrainPatchChecker trainPatchChecker;

} /* namespace cnn */

} /* namespace gml */
