/*
 * FindThreshold.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#include "FindThreshold.h"

namespace gml {

namespace nn {

BeginPropertyDefinitions(FindThreshold)

  ReflectableBase(DefaultWorkflowElement<FindThreshold>)

  WorkflowProperty(InitialModel, Input("NN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(PatchWidth)
  WorkflowProperty(PatchHeight)
  WorkflowProperty(PatchDepth)
  WorkflowProperty(PatchCounts, Description("Number of simultaneously processed patches in x, y, and z direction. (Must have 3 values)"))
  WorkflowProperty(Model, Output("PNN"))

EndPropertyDefinitions

FindThreshold::FindThreshold() : _PatchWidth(0), _PatchHeight(0), _PatchDepth(0) {
  setLabel("Tresh");
}

FindThresholdChecker findThresholdChecker;

} /* namespace nn */

} /* namespace gml */
