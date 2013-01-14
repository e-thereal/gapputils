/*
 * TestModule.cpp
 *
 *  Created on: 2012-10-20
 *      Author: tombr
 */

#include "TestModule.h"

#include <capputils/DeprecatedAttribute.h>

using namespace capputils::attributes;

namespace gapputils {
namespace testing {

BeginPropertyDefinitions(TestModule, Deprecated("Use only for testing purpose."))

  ReflectableBase(workflow::DefaultWorkflowElement<TestModule>)

  WorkflowProperty(Input, Input("In"))
  WorkflowProperty(Output, Output("Out"), Volatile())
  WorkflowProperty(Cycles)
  WorkflowProperty(Delay, Description("Duration of a cycle in ms."))

EndPropertyDefinitions

TestModule::TestModule() : _Cycles(10), _Delay(100) {
  setLabel("Test");
}

TestModule::~TestModule() {
}

void TestModule::update(workflow::IProgressMonitor* monitor) const {
  for (int i = 0; i < getCycles() && (monitor ? !monitor->getAbortRequested() : 1); ++i) {
#ifndef WIN32
    usleep(1000 * getDelay());
#endif
    if (monitor)
      monitor->reportProgress(100. * i / getCycles());
  }
  newState->setOutput(getInput());
}

} /* namespace testing */
} /* namespace gapputils */
