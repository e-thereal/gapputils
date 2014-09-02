/*
 * DummyLoop.cpp
 *
 *  Created on: Sep 2, 2014
 *      Author: tombr
 */

#include "DummyLoop.h"

#include <capputils/attributes/DeprecatedAttribute.h>

namespace debug {

#ifdef _RELEASE
BeginPropertyDefinitions(DummyLoop, Deprecated("Only available in debug mode."))
#else
BeginPropertyDefinitions(DummyLoop)
#endif

  ReflectableBase(DefaultWorkflowElement<DummyLoop>)

  WorkflowProperty(Iterations)
  WorkflowProperty(ShowProgress, Flag())

EndPropertyDefinitions

DummyLoop::DummyLoop() : _Iterations(10000), _ShowProgress(true) {
  setLabel("Loop");
}

void DummyLoop::update(IProgressMonitor* monitor) const {
  for (int i = 0; i < getIterations(); ++i) {
    if (monitor && getShowProgress())
      monitor->reportProgress((double)i * 100. / getIterations());
  }
}

} /* namespace debug */
