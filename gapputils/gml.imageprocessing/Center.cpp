/*
 * Center.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: tombr
 */

#include "Center.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Center)

  ReflectableBase(DefaultWorkflowElement<Center>)

  WorkflowProperty(Input, Input("I"), NotNull<Type>())
  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(RoundToNearest, Flag())
  WorkflowProperty(Transform, Output("M"))

EndPropertyDefinitions

Center::Center() : _RoundToNearest(true) {
  setLabel("CoG");
}

} /* namespace imageprocessing */

} /* namespace gml */
