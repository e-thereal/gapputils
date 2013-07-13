/*
 * Inference.cpp
 *
 *  Created on: Jul 12, 2013
 *      Author: tombr
 */

#include "Inference.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Inference)

  ReflectableBase(DefaultWorkflowElement<Inference>)

  WorkflowProperty(Model, Input("DBM"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Direction)
  WorkflowProperty(GpuCount)
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

Inference::Inference() : _GpuCount(1) {
  setLabel("Inference");
}

InferenceChecker inferenceChecker;

} /* namespace convrbm4d */

} /* namespace gml */
