/*
 * WorkflowInterface.cpp
 *
 *  Created on: May 24, 2011
 *      Author: tombr
 */

#include "WorkflowInterface.h"

#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>

#include <gapputils/attributes/LabelAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace workflow {

int WorkflowInterface::LabelId = -1;

BeginPropertyDefinitions(WorkflowInterface)
  DefineProperty(Label, Label(), Observe(LabelId = Id))
EndPropertyDefinitions

WorkflowInterface::WorkflowInterface() : _Label("Interface") {

}

WorkflowInterface::~WorkflowInterface() {
}

}

}
