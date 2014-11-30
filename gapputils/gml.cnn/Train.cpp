/*
 * Train.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

namespace gml {

namespace cnn {


BeginPropertyDefinitions(Train)

  ReflectableBase(DefaultWorkflowElement<Train>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(InitialModel, Input("CNN"), NotNull<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(TrialEpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(FilterBatchSize)

  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(LearningRates, NotEmpty<Type>())
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(WeightCosts)
  WorkflowProperty(InitialWeights, Description("If given, these weights will be tested as initial weights and will override the initial weights."))
  WorkflowProperty(RandomizeTraining, Flag())
  WorkflowProperty(Model, Output("CNN"))

EndPropertyDefinitions

Train::Train() : _EpochCount(100), _TrialEpochCount(20), _BatchSize(50), _LearningDecay(50), _WeightCosts(0.0002), _RandomizeTraining(true)
{
  setLabel("Train");
  _LearningRates.push_back(0.0001);
}

TrainChecker trainChecker;

} /* namespace cnn */

} /* namespace gml */
