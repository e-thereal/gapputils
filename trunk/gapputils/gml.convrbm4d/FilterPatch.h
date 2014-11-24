/*
 * Filter.h
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#ifndef GML_FILTERPATCH_H_
#define GML_FILTERPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"
#include "CodingDirection.h"

namespace gml {

namespace convrbm4d {

struct FilterPatchChecker { FilterPatchChecker(); };

class FilterPatch : public DefaultWorkflowElement<FilterPatch> {

  friend class FilterPatchChecker;

public:
  typedef model_t::host_tensor_t host_tensor_t;
  typedef model_t::value_t value_t;

  InitReflectableClass(FilterPatch)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Direction, CodingDirection)

  Property(SuperPatchWidth, int)
  Property(SuperPatchHeight, int)
  Property(SuperPatchDepth, int)
  Property(FilterBatchSize, int)
  Property(GpuCount, int)
  Property(DoubleWeights, bool)
  Property(OnlyFilters, bool)
  Property(SampleUnits, bool)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)

public:
  FilterPatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_FILTERPATCH_H_ */
