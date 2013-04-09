/*
 * SplitModel.h
 *
 *  Created on: Jan 9, 2013
 *      Author: tombr
 */

#ifndef GML_SPLITMODEL_H_
#define GML_SPLITMODEL_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class SplitModel : public DefaultWorkflowElement<SplitModel> {
  typedef Model::value_t value_t;
  typedef Model::tensor_t tensor_t;
  typedef Model::dim_t dim_t;

  InitReflectableClass(SplitModel)

  Property(Model, boost::shared_ptr<Model>)

  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(VisibleBias, boost::shared_ptr<tensor_t>)
  Property(HiddenBiases, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(FilterKernelSize, dim_t)
  Property(Mean, double)
  Property(Stddev, double)
  Property(VisibleUnitType, UnitType)
  Property(HiddenUnitType, UnitType)

public:
  SplitModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* SPLITMODEL_H_ */
