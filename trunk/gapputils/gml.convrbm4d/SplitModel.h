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
  typedef model_t::value_t value_t;
  typedef model_t::host_tensor_t tensor_t;
  typedef model_t::v_host_tensor_t v_tensor_t;
  typedef model_t::dim_t dim_t;

  InitReflectableClass(SplitModel)

  Property(Model, boost::shared_ptr<model_t>)
  Property(MaxFilterCount, int)

  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(VisibleBias, boost::shared_ptr<tensor_t>)
  Property(HiddenBiases, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(FilterKernelSize, dim_t)
  Property(Mean, double)
  Property(Stddev, double)
  Property(VisibleUnitType, tbblas::deeplearn::unit_type)
  Property(HiddenUnitType, tbblas::deeplearn::unit_type)

public:
  SplitModel();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* GML_SPLITMODEL_H_ */
