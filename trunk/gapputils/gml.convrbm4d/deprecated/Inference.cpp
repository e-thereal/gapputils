/*
 * Inference.cpp
 *
 *  Created on: Jul 12, 2013
 *      Author: tombr
 */

#include "Inference.h"
#include <capputils/DeprecatedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Inference, Deprecated("Use gml.dbm.Inference instead."))

  ReflectableBase(DefaultWorkflowElement<Inference>)

  WorkflowProperty(Model, Input("DBM"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(Iterations)
  WorkflowProperty(GpuCount)
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

Inference::Inference() : _Iterations(1), _GpuCount(1) {
  setLabel("Inference");
}

InferenceChecker inferenceChecker;

} /* namespace convrbm4d */

} /* namespace gml */
