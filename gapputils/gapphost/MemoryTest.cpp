/*
 * MemoryTest.cpp
 *
 *  Created on: Apr 5, 2013
 *      Author: tombr
 */

#include "MemoryTest.h"

#include <cstdlib>

namespace debug {

BeginPropertyDefinitions(MemoryTest)

  ReflectableBase(DefaultWorkflowElement<MemoryTest>)

  WorkflowProperty(Input, Input(""))
  WorkflowProperty(Size)
  WorkflowProperty(Iterations)
  WorkflowProperty(Delay)
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

MemoryTest::MemoryTest() : _Size(1), _Iterations(5), _Delay(1) {
  setLabel("MemTest");
}

void MemoryTest::update(IProgressMonitor* monitor) const {
  for (int i = 0; i < getIterations() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    _sleep(getDelay());
    if (monitor)
      monitor->reportProgress(100.0 * i / getIterations());
  }

  newState->setOutput(boost::make_shared<std::vector<double> >(getSize()));
}

} /* namespace debug */
