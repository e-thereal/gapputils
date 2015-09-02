/*
 * VisualizeFilters.h
 *
 *  Created on: Aug 4, 2015
 *      Author: tombr
 */

#ifndef GML_VISUALIZEFILTERS_H_
#define GML_VISUALIZEFILTERS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace encoder {

class VisualizeFilters : public DefaultWorkflowElement<VisualizeFilters> {

  InitReflectableClass(VisualizeFilters)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Layer, int)
  Property(FilterBatchLength, std::vector<int>)
  Property(SubRegionCount, host_tensor_t::dim_t)

  Property(EncodingFilters, boost::shared_ptr<v_host_tensor_t>)
  Property(DecodingFilters, boost::shared_ptr<v_host_tensor_t>)

public:
  VisualizeFilters();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_VISUALIZEFILTERS_H_ */
