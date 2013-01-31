/*
 * VerticalAnnotation.cpp
 *
 *  Created on: Jan 31, 2013
 *      Author: tombr
 */

#include "VerticalAnnotation.h"

namespace interfaces {

BeginPropertyDefinitions(VerticalAnnotation)

  ReflectableBase(DefaultWorkflowElement<VerticalAnnotation>)

EndPropertyDefinitions

VerticalAnnotation::VerticalAnnotation() {
  setLabel("Annotation");
}

} /* namespace interfaces */
