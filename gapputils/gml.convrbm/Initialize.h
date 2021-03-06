/*
 * Initialize.h
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm {

struct InitializeChecker { InitializeChecker(); };

class Initialize : public DefaultWorkflowElement<Initialize> {
public:
  typedef Model::value_t value_t;
  typedef Model::tensor_t tensor_t;

  friend class InitializeChecker;

  InitReflectableClass(Initialize)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(FilterWidth, int)
  Property(FilterHeight, int)
  Property(FilterCount, int)
  int dummy;
  Property(Sigma, double)
  Property(WeightMean, double)
  Property(WeightStddev, double)
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)

  Property(Model, boost::shared_ptr<Model>)

public:
  Initialize();
  virtual ~Initialize();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm */
} /* namespace gml */
#endif /* INITIALIZE_H_ */
