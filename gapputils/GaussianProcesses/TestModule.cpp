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
    DefineProperty(X1, Input(), Observe(PROPERTY_ID))
    DefineProperty(X2, Input(), Observe(PROPERTY_ID))
    DefineProperty(Y1, Output(), Observe(PROPERTY_ID))
    DefineProperty(Y2, Output(), Observe(PROPERTY_ID))

EndPropertyDefinitions

TestModule::TestModule() : _X1(0), _X2(0), _Y1(0), _Y2(0), upToDate(true) {
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
#ifdef WIN32
    _sleep(100);
#else
    usleep(10000);
#endif
  }
  result = getX1() * getX2();
}

void TestModule::writeResults() {
  setY1(result);
  setY2(result);
}

}
