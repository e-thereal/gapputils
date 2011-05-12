#pragma once
#ifndef _GAPPUTILS_DEFAULTINTERFACE_H_
#define _GAPPUTILS_DEFAULTINTERFACE_H_

#include <WorkflowElement.h>

namespace gapputils {

namespace host {

class DefaultInterface : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(DefaultInterface)

public:
  DefaultInterface(void);
  virtual ~DefaultInterface(void);

  virtual void execute(workflow::IProgressMonitor* monitor) const { }
  virtual void writeResults() { }
};

}

}

#endif
