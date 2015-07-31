/*
 * Transform.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: tombr
 */

#include "Transform.h"

namespace gml {
namespace imageprocessing {

BeginPropertyDefinitions(Transform)

  ReflectableBase(DefaultWorkflowElement<Transform>)

  WorkflowProperty(Input, Input("I"), NotNull<Type>())
  WorkflowProperty(Transform, Input("M"), NotNull<Type>())
  WorkflowProperty(Output, Output("I"))


EndPropertyDefinitions

Transform::Transform() {
  setLabel("Trans");
}

} /* namespace imageprocessing */

} /* namespace gml */
