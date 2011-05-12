/*
 * WorkflowElement.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowElement.h"

#include "LabelAttribute.h"
#include <ObserveAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

namespace workflow {

BeginAbstractPropertyDefinitions(WorkflowElement)

    DefineProperty(Label, Label(), Observe(PROPERTY_ID))

EndPropertyDefinitions

WorkflowElement::WorkflowElement() : _Label("Label") {
}

}

}
