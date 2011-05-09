#pragma once
#ifndef _GAPPUTILS_DEFAULTINTERFACE_H_
#define _GAPPUTILS_DEFAULTINTERFACE_H_

#include <ReflectableClass.h>

namespace gapputils {

namespace host {

class DefaultInterface : public capputils::reflection::ReflectableClass
{
  InitReflectableClass(DefaultInterface)

public:
  DefaultInterface(void);
  virtual ~DefaultInterface(void);
};

}

}

#endif