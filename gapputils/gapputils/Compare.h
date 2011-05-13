#pragma once

#ifndef _GAPPUTILS_H_
#define _GAPPUTILS_H_

#include "gapputils.h"
#include "WorkflowElement.h"
#include <Enumerators.h>

namespace gapputils {

ReflectableEnum(ErrorType, MSE, SSD, CC);

class Compare : public workflow::WorkflowElement
{

  InitReflectableClass(Compare)

  Property(Type, ErrorType)
  Property(X, double*)
  Property(Y, double*)
  Property(Count, int)
  Property(Error, double)

private:
  mutable Compare* data;

public:
  Compare(void);
  ~Compare(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif
