/*
 * TestInterface.cpp
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#include "TestInterface.h"

#include <OutputAttribute.h>
#include <ObserveAttribute.h>
#include <VolatileAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(TestInterface)

 DefineProperty(Pdf, Output(), Observe(PROPERTY_ID), Volatile())

EndPropertyDefinitions

TestInterface::TestInterface() {

}

TestInterface::~TestInterface() {
}

}
