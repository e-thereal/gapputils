/*
 * MemoryTest.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: tombr
 */

#include "MemoryTest.h"

#include <capputils/attributes/DeprecatedAttribute.h>
#include <gapputils/attributes/GroupAttribute.h>

#include <cstdlib>

namespace debug {

#ifdef _RELEASE
BeginPropertyDefinitions(MemoryTest, Deprecated("Only available in debug mode."))
#else
BeginPropertyDefinitions(MemoryTest)
#endif

  ReflectableBase(DefaultWorkflowElement<MemoryTest>)

  WorkflowProperty(Input, Input("In"))
  WorkflowProperty(Size, Group("Other parameter"))
  WorkflowProperty(Iterations, Group("Other parameter"))
  WorkflowProperty(Delay, Group("Other parameter"))
  WorkflowProperty(Output, Output("Out"))

EndPropertyDefinitions

MemoryTest::MemoryTest() : _Size(1), _Iterations(5), _Delay(1) {
  setLabel("MemTest");
}

void MemoryTest::update(IProgressMonitor* monitor) const {
  for (int i = 0; i < getIterations() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
#ifdef WIN32
    _sleep(getDelay());
#else
    sleep(getDelay());
#endif
    if (monitor)
      monitor->reportProgress(100.0 * (i + 1) / getIterations());
  }

  newState->setOutput(boost::make_shared<std::vector<double> >(getSize()));
}

} /* namespace debug */
