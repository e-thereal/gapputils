/*
 * WorkflowInterface.cpp
 *
 *  Created on: May 24, 2011
 *      Author: tombr
 */

#include "WorkflowInterface.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>

#include "HideAttribute.h"
#include "LabelAttribute.h"

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
