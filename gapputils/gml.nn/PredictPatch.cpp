/*
 * PredictPatch.cpp
 *
 *  Created on: Dec 06, 2014
 *      Author: tombr
 */

#include "PredictPatch.h"

namespace gml {

namespace nn {

BeginPropertyDefinitions(PredictPatch)

  ReflectableBase(DefaultWorkflowElement<PredictPatch>)

  WorkflowProperty(Model, Input("NN"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(PatchWidth)
  WorkflowProperty(PatchHeight)
  WorkflowProperty(PatchDepth)
  WorkflowProperty(PatchCounts, Description("Number of simultaneously processed patches in x, y, and z direction. (Must have 3 values)"))
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

PredictPatch::PredictPatch() : _PatchWidth(16), _PatchHeight(16), _PatchDepth(16) {
  setLabel("Predict");
}

PredictPatchChecker predictPatchChecker;

} /* namespace nn */

} /* namespace gml */
