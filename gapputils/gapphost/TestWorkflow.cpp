/*
 * TestWorkflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "TestWorkflow.h"

#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/ObserveAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

BeginPropertyDefinitions(TestWorkflow)

  DefineProperty(Name, Input(), Observe(PROPERTY_ID))
  DefineProperty(In1, Input(), Observe(PROPERTY_ID))
  DefineProperty(Out1, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

TestWorkflow::TestWorkflow() {
}

TestWorkflow::~TestWorkflow() {
}

}
