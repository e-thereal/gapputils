/*
 * StackTensors.cpp
 *
 *  Created on: Jun 24, 2013
 *      Author: tombr
 */

#include "StackTensors.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(StackTensors)

  ReflectableBase(DefaultWorkflowElement<StackTensors>)

  WorkflowProperty(Tensor1, Input("T1"), NotNull<Type>())
  WorkflowProperty(Tensor2, Input("T2"), NotNull<Type>())
  WorkflowProperty(OutputTensor, Input("Out"))

EndPropertyDefinitions

StackTensors::StackTensors() {
  setLabel("Stack");
}

void StackTensors::update(IProgressMonitor* monitor) const {

}

} /* namespace convrbm4d */

} /* namespace gml */
