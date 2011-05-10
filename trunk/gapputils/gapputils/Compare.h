#pragma once

#ifndef _GAPPUTILS_H_
#define _GAPPUTILS_H_

#include "gapputils.h"
#include <ReflectableClass.h>
#include <ObservableClass.h>
#include <Enumerators.h>

namespace gapputils {

ReflectableEnum(ErrorType, MSE, SSD, CC);

class Compare : public capputils::reflection::ReflectableClass,
                public capputils::ObservableClass
{

  InitReflectableClass(Compare)

  Property(Type, ErrorType)
  Property(X, double*)
  Property(Y, double*)
  Property(Count, int)
  Property(Error, double)

public:
  Compare(void);
  ~Compare(void);

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif