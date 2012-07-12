/*
 * WorkflowElement.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowElement.h"

#include <capputils/NoParameterAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>

#include "HideAttribute.h"
#include "LabelAttribute.h"
#include "Logbook.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace workflow {

int WorkflowElement::labelId;

BeginAbstractPropertyDefinitions(WorkflowElement)

  DefineProperty(Label, Label(), Observe(labelId = PROPERTY_ID))
  DefineProperty(HostInterface, Volatile(), Hide(), NoParameter(), Observe(PROPERTY_ID))

EndPropertyDefinitions

WorkflowElement::WorkflowElement() : _Label("Element"), logbook(new Logbook()) {
}

Logbook& WorkflowElement::getLogbook() const {
  return *logbook;
}

}

}
