#pragma once

#ifndef GAPPUTILS_COMMON_COMPARE_H_
#define GAPPUTILS_COMMON_COMPARE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace common {

CapputilsEnumerator(ErrorType, MSE, SE, RSE);

class Compare : public workflow::DefaultWorkflowElement<Compare>
{

  InitReflectableClass(Compare)

  Property(X, boost::shared_ptr<std::vector<double> >)
  Property(Y, boost::shared_ptr<std::vector<double> >)
  Property(Type, ErrorType)
  Property(Error, double)

public:
  Compare(void);
  ~Compare(void);

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_COMMON_COMPARE_H_ */
