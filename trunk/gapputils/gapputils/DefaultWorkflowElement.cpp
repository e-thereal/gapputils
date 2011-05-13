/*
 * DefaultWorkflowElement.cpp
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#include "DefaultWorkflowElement.h"

namespace gapputils {

namespace workflow {

BeginPropertyDefinitions(DefaultWorkflowElement)

  ReflectableBase(WorkflowElement)

EndPropertyDefinitions

DefaultWorkflowElement::DefaultWorkflowElement() {

}

DefaultWorkflowElement::~DefaultWorkflowElement() {
}

}

}
