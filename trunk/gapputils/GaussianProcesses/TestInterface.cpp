/*
 * TestInterface.cpp
 *
 *  Created on: May 13, 2011
 *      Author: tombr
 */

#include "TestInterface.h"

#include <capputils/OutputAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/VolatileAttribute.h>

using namespace capputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(TestInterface)

 DefineProperty(Pdf, Output(), Observe(PROPERTY_ID), Volatile())

EndPropertyDefinitions

TestInterface::TestInterface() {

}

TestInterface::~TestInterface() {
}

}
