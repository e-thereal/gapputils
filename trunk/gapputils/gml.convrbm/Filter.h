/*
 * Filter.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef FILTER_H_
#define FILTER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"
#include "CodingDirection.h"

namespace gml {
namespace convrbm {

struct FilterChecker { FilterChecker(); };

class Filter : public DefaultWorkflowElement<Filter> {

  friend class FilterChecker;

public:
  typedef Model::tensor_t host_tensor_t;
  typedef Model::value_t value_t;

  InitReflectableClass(Filter)

  Property(Model, boost::shared_ptr<Model>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Direction, CodingDirection)
  Property(GpuCount, int)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)

public:
  Filter();
  virtual ~Filter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm */
} /* namespace gml */
#endif /* FILTER_H_ */
