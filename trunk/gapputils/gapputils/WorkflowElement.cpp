/*
 * WorkflowElement.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowElement.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/VolatileAttribute.h>

#include "HideAttribute.h"
#include "LabelAttribute.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace workflow {

int WorkflowElement::labelId;

BeginAbstractPropertyDefinitions(WorkflowElement)

  DefineProperty(Label, Label(), Observe(labelId = PROPERTY_ID))
  DefineProperty(SetOnCompilation, TimeStamp(PROPERTY_ID), Hide(), Volatile())

EndPropertyDefinitions

WorkflowElement::WorkflowElement() : _Label("Element") {
}

}

}
