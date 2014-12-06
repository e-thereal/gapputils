/*
 * TrainPatch.cpp
 *
 *  Created on: Oct 14, 2014
 *      Author: tombr
 */

#include "TrainPatch.h"

namespace gml {

namespace nn {


BeginPropertyDefinitions(TrainPatch)

  ReflectableBase(DefaultWorkflowElement<TrainPatch>)

  WorkflowProperty(InitialModel, Input("NN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mask, Input("M"))
  WorkflowProperty(PatchWidth)
  WorkflowProperty(PatchHeight)
  WorkflowProperty(PatchDepth)
  WorkflowProperty(PatchCount)
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(PositiveRatio)

  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(LearningRate)
  WorkflowProperty(WeightCosts)
  WorkflowProperty(RandomizeTraining, Flag())
  WorkflowProperty(Model, Output("NN"))
  WorkflowProperty(Patches, Output("Ps"))
  WorkflowProperty(Targets, Output("Ts"))
  WorkflowProperty(Predictions, Output("Ps"))

EndPropertyDefinitions

TrainPatch::TrainPatch() : _PatchWidth(16), _PatchHeight(16), _PatchDepth(16), _PatchCount(16), _EpochCount(100), _BatchSize(50), _PositiveRatio(0.5), _LearningRate(0.0001), _WeightCosts(0.0002), _RandomizeTraining(true) {
  setLabel("TrainPatch");
}

TrainPatchChecker trainPatchChecker;

} /* namespace nn */

} /* namespace gml */
