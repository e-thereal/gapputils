/*
 * WorkflowElement.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowElement.h"

#include <capputils/Logbook.h>

#include <capputils/attributes/NoParameterAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/TimeStampAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>

#include <gapputils/attributes/LabelAttribute.h>

using namespace capputils;
using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace workflow {

int WorkflowElement::labelId;

BeginAbstractPropertyDefinitions(WorkflowElement)

  DefineProperty(Label, Label(), Observe(labelId = Id))

EndPropertyDefinitions

WorkflowElement::WorkflowElement() : _Label("Element"), logbook(new Logbook()), atomicWorkflow(false) {
}

Logbook& WorkflowElement::getLogbook() const {
  return *logbook;
}

boost::shared_ptr<IGapphostInterface> WorkflowElement::getHostInterface() const {
  return hostInterface;
}

void WorkflowElement::setHostInterface(const boost::shared_ptr<IGapphostInterface>& interface) {
  hostInterface = interface;
}

bool WorkflowElement::getAtomicWorkflow() const {
  return atomicWorkflow;
}

void WorkflowElement::setAtomicWorkflow(bool atomic) {
  atomicWorkflow = atomic;
}

}

}
