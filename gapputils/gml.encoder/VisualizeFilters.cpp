/*
 * VisualizeFilters.cpp
 *
 *  Created on: Aug 4, 2015
 *      Author: tombr
 */

#include "VisualizeFilters.h"

namespace gml {

namespace encoder {

BeginPropertyDefinitions(VisualizeFilters)

  ReflectableBase(DefaultWorkflowElement<VisualizeFilters>)

  WorkflowProperty(Model, Input("Enn"), NotNull<Type>())
  WorkflowProperty(Layer, Description("A value of -1 means use the top-most layer."))
  WorkflowProperty(FilterBatchLength)
  WorkflowProperty(SubRegionCount, Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(EncodingFilters, Output("EF"))
  WorkflowProperty(DecodingFilters, Output("DF"))

EndPropertyDefinitions

VisualizeFilters::VisualizeFilters() : _Layer(-1), _SubRegionCount(tbblas::seq<host_tensor_t::dimCount>(1)) {
  setLabel("Filters");
}

} /* namespace encoder */

} /* namespace gml */
