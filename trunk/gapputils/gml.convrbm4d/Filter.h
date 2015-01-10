/*
 * Filter.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_FILTER_H_
#define GML_FILTER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"
#include "CodingDirection.h"

namespace gml {

namespace convrbm4d {

struct FilterChecker { FilterChecker(); };

class Filter : public DefaultWorkflowElement<Filter> {

  friend class FilterChecker;

public:
  typedef model_t::host_tensor_t host_tensor_t;
  typedef model_t::value_t value_t;

  InitReflectableClass(Filter)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Direction, CodingDirection)
  Property(FilterBatchSize, int)
  Property(SubRegionCount, host_tensor_t::dim_t)
  Property(DoubleWeights, bool)
  Property(OnlyFilters, bool)
  Property(SampleUnits, bool)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)

public:
  Filter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_FILTER_H_ */
