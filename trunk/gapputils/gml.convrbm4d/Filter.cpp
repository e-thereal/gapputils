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
  WorkflowProperty(Outputs, Output("Ts"))
EndPropertyDefinitions

Filter::Filter() : _GpuCount(1) {
  setLabel("Filter");
}

FilterChecker filterChecker;

} /* namespace convrbm4d */

} /* namespace gml */
