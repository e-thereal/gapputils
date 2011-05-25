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

BeginPropertyDefinitions(WorkflowInterface)
  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(SetOnCompilation, TimeStamp(PROPERTY_ID), Hide(), Volatile())
EndPropertyDefinitions

WorkflowInterface::WorkflowInterface() : _Label("Interface") {

}

WorkflowInterface::~WorkflowInterface() {
}

}

}
