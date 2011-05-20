#pragma once
#ifndef _GAPPUTILS_DEFAULTINTERFACE_H_
#define _GAPPUTILS_DEFAULTINTERFACE_H_

#include <DefaultWorkflowElement.h>

namespace gapputils {

namespace host {

class DefaultInterface : public gapputils::workflow::DefaultWorkflowElement
{
  InitReflectableClass(DefaultInterface)

public:
  DefaultInterface(void);
  virtual ~DefaultInterface(void);
};

}

}

#endif
