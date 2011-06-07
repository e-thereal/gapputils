#pragma once

#ifndef _GAPPUTILS_H_
#define _GAPPUTILS_H_

#include <gapputils/WorkflowElement.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace common {

ReflectableEnum(ErrorType, MSE, SE, RSE);

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

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

}

#endif
