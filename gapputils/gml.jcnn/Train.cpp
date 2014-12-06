/*
 * Train.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "Train.h"

namespace gml {

namespace jcnn {


BeginPropertyDefinitions(Train)

  ReflectableBase(DefaultWorkflowElement<Train>)

  WorkflowProperty(InitialModel, Input("JCNN"), NotNull<Type>())
  WorkflowProperty(LeftTrainingSet, Input("LD"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(RightTrainingSet, Input("RD"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LeftFilterBatchSize)
  WorkflowProperty(RightFilterBatchSize)

  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(CLearningRate)
  WorkflowProperty(DLearningRate)
  WorkflowProperty(WeightCosts)
  WorkflowProperty(RandomizeTraining, Flag())
  WorkflowProperty(Model, Output("JCNN"))

EndPropertyDefinitions

Train::Train() : _EpochCount(100), _BatchSize(50), _CLearningRate(0.0001), _DLearningRate(0.0001),
                 _WeightCosts(0.0002), _RandomizeTraining(true)
{
  setLabel("Train");
}

TrainChecker trainChecker;

} /* namespace cnn */

} /* namespace gml */
