#pragma once
#ifndef _GAPPUTILS_DEFAULTINTERFACE_H_
#define _GAPPUTILS_DEFAULTINTERFACE_H_

#include <gapputils/WorkflowInterface.h>

namespace gapputils {

namespace host {

class DefaultInterface : public gapputils::workflow::WorkflowInterface
{
  InitReflectableClass(DefaultInterface)

public:
  DefaultInterface(void);
  virtual ~DefaultInterface(void);
};

}

}

#endif
