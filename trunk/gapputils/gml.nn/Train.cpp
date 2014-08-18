/*
 * Train.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

namespace gml {

namespace nn {


BeginPropertyDefinitions(Train)

  ReflectableBase(DefaultWorkflowElement<Train>)

  WorkflowProperty(InitialModel, Input("NN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)

  WorkflowProperty(LearningRate)
  WorkflowProperty(WeightCosts)
  WorkflowProperty(ShuffleTrainingSet, Flag())
  WorkflowProperty(Model, Output("NN"))

EndPropertyDefinitions

Train::Train() : _EpochCount(100), _BatchSize(50), _LearningRate(0.0001), _WeightCosts(0.0002), _ShuffleTrainingSet(true) {
  setLabel("Train");
}

TrainChecker trainChecker;

} /* namespace nn */

} /* namespace gml */
