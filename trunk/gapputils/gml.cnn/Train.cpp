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
  WorkflowProperty(BatchSize)
  WorkflowProperty(FilterBatchSize)

  WorkflowProperty(CLearningRate)
  WorkflowProperty(DLearningRate)
  WorkflowProperty(WeightCosts)
  WorkflowProperty(RandomizeTraining, Flag())
  WorkflowProperty(Model, Output("CNN"))

EndPropertyDefinitions

Train::Train() : _EpochCount(100), _BatchSize(50), _CLearningRate(0.0001), _DLearningRate(0.0001),
                 _WeightCosts(0.0002), _RandomizeTraining(true)
{
  setLabel("Train");
}

TrainChecker trainChecker;

} /* namespace cnn */

} /* namespace gml */
