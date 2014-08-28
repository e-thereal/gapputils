#pragma once

#ifndef GML_COMPARE_H_
#define GML_COMPARE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <capputils/Enumerators.h>

namespace gml {

namespace core {

CapputilsEnumerator(ErrorType, MSE, RMSE, RRMSE, Multinomial);

class Compare : public DefaultWorkflowElement<Compare> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Compare)

  Property(X, boost::shared_ptr<data_t>)
  Property(Y, boost::shared_ptr<data_t>)
  Property(Xs, boost::shared_ptr<v_data_t>)
  Property(Ys, boost::shared_ptr<v_data_t>)
  Property(Type, ErrorType)
  Property(Error, double)

public:
  Compare();

protected:
  virtual void update(workflow::IProgressMonitor* monitor) const;
};

}

}

#endif /* GAPPUTILS_COMMON_COMPARE_H_ */
