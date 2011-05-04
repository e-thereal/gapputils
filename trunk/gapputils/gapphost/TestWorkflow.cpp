/*
 * TestWorkflow.cpp
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#include "TestWorkflow.h"

#include <InputAttribute.h>
#include <OutputAttribute.h>
#include <ObserveAttribute.h>

using namespace capputils::attributes;

namespace gapputils {

using namespace attributes;

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
