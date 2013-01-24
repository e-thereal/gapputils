#pragma once

#ifndef GML_COMPARE_H_
#define GML_COMPARE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <capputils/Enumerators.h>

namespace gml {

namespace core {

CapputilsEnumerator(ErrorType, MSE, RMSE, RRMSE);

class Compare : public DefaultWorkflowElement<Compare>
{

  InitReflectableClass(Compare)

  Property(X, boost::shared_ptr<std::vector<double> >)
  Property(Y, boost::shared_ptr<std::vector<double> >)
  Property(Type, ErrorType)
  Property(Error, double)

public:
  Compare(void);

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_COMMON_COMPARE_H_ */
