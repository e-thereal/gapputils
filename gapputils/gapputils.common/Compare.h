#pragma once

#ifndef GAPPUTILS_COMMON_COMPARE_H_
#define GAPPUTILS_COMMON_COMPARE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace common {

ReflectableEnum(ErrorType, MSE, SE, RSE);

class Compare : public workflow::DefaultWorkflowElement<Compare>
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

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_COMMON_COMPARE_H_ */
