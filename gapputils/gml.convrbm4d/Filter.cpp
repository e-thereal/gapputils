/*
 * Filter.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "Filter.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Filter)
  ReflectableBase(DefaultWorkflowElement<Filter>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(GpuCount)
  WorkflowProperty(DoubleWeights, Flag())
  WorkflowProperty(OnlyFilters, Flag())
  WorkflowProperty(SampleUnits, Flag())
  WorkflowProperty(Outputs, Output("Ts"))
EndPropertyDefinitions

Filter::Filter() : _GpuCount(1), _DoubleWeights(false), _OnlyFilters(false), _SampleUnits(false) {
  setLabel("Filter");
}

FilterChecker filterChecker;

} /* namespace convrbm4d */

} /* namespace gml */
