/*
 * PredictPatch.cpp
 *
 *  Created on: Dec 4, 2014
 *      Author: tombr
 */

#include "PredictPatch.h"

namespace gml {

namespace cnn {

BeginPropertyDefinitions(PredictPatch)

  ReflectableBase(DefaultWorkflowElement<PredictPatch>)

  WorkflowProperty(Model, Input("CNN"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(FilterBatchSize)
  WorkflowProperty(PatchCounts, Description("Number of patches in x-, y-, and z-direction used for the multi-patch training."))
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

PredictPatch::PredictPatch() {
  setLabel("Predict");
}

PredictPatchChecker predictPatchChecker;

} /* namespace cnn */

} /* namespace gml */
