/*
 * HorizontalAnnotation.cpp
 *
 *  Created on: Jan 31, 2013
 *      Author: tombr
 */

#include "HorizontalAnnotation.h"

namespace interfaces {

BeginPropertyDefinitions(HorizontalAnnotation)

  ReflectableBase(DefaultWorkflowElement<HorizontalAnnotation>)

EndPropertyDefinitions

HorizontalAnnotation::HorizontalAnnotation() {
  setLabel("Annotation");
}

} /* namespace interfaces */
