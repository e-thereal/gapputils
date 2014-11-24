/*
 * Infer.cpp
 *
 *  Created on: Oct 28, 2014
 *      Author: tombr
 */

#include "Infer.h"

#include <capputils/attributes/IsNullAttribute.h>
#include <capputils/attributes/OrAttribute.h>

namespace gml {

namespace dbn {

BeginPropertyDefinitions(Infer)

  ReflectableBase(DefaultWorkflowElement<Infer>)

  WorkflowProperty(Model, Input("Dbn"), NotNull<Type>())
  WorkflowProperty(InputTensors, Input("Ts"), Or(IsNull<Type>(), NotEmpty<Type>()))
  WorkflowProperty(InputUnits, Input("Us"), Or(IsNull<Type>(), NotEmpty<Type>()))
  WorkflowProperty(Layer, Description("Specifies the hidden layer to be inferred. A value of 0 indicates the visible units. A value of -1 indicates the top-most layer."))
  WorkflowProperty(TopDown, Flag(), Description("Performs top-down inference if set and bottom-up inference otherwise."))
  WorkflowProperty(GpuCount)
  WorkflowProperty(FilterBatchLength)
  WorkflowProperty(OutputTensors, Output("Ts"))
  WorkflowProperty(OutputUnits, Output("Us"))

EndPropertyDefinitions

Infer::Infer() : _Layer(-1), _TopDown(false), _GpuCount(1) {
  setLabel("Infer");
}

InferChecker inferChecker;

} /* namespace dbn */

} /* namespace gml */
