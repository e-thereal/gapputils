/*
 * TestModule.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "TestModule.h"

#include <InputAttribute.h>
#include <OutputAttribute.h>
#include <ObserveAttribute.h>

#include <EventHandler.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace GaussianProcesses {

BeginPropertyDefinitions(TestModule)

    ReflectableBase(gapputils::workflow::WorkflowElement)
    DefineProperty(X, Input(), Observe(PROPERTY_ID))
    DefineProperty(Y, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

TestModule::TestModule() : _X(0), _Y(0), upToDate(true) {
  setLabel("TestModule");
  Changed.connect(capputils::EventHandler<TestModule>(this, &TestModule::changedHandler));
}

TestModule::~TestModule() {
}

void TestModule::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == 1) {
    upToDate = false;
  }
}

void TestModule::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  for (int i = 0; i <= 100; ++i) {
    monitor->reportProgress(i);
    usleep(10000);
  }
  result = 2 * getX();
}

void TestModule::writeResults() {
  setY(result);
}

}
