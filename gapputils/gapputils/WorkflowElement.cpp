/*
 * WorkflowElement.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowElement.h"

#include "LabelAttribute.h"
#include <ObserveAttribute.h>
#include <TimeStampAttribute.h>
#include <VolatileAttribute.h>
#include "HideAttribute.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

using namespace attributes;

namespace workflow {

BeginAbstractPropertyDefinitions(WorkflowElement)

  DefineProperty(Label, Label(), Observe(PROPERTY_ID))
  DefineProperty(SetOnCompilation, TimeStamp(PROPERTY_ID), Hide(), Volatile())

EndPropertyDefinitions

WorkflowElement::WorkflowElement() : _Label("Label") {
}

}

}
