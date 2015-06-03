/*
 * Predict.cpp
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#include "Predict.h"

#include <gapputils/attributes/GroupAttribute.h>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(Predict)

  ReflectableBase(DefaultWorkflowElement<Predict>)

  WorkflowProperty(MaximumLayer, Description("A value of -1 indicates the output layer."))
  WorkflowProperty(CalculateDeltas, Flag())
  WorkflowProperty(Objective, Enumerator<Type>())
  WorkflowProperty(SensitivityRatio)
  WorkflowProperty(FilterBatchSize, Group("Performance"))
  WorkflowProperty(SubRegionCount, Group("Performance"), Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(Model, Input("ENN"), NotNull<Type>(), Group("Input/output"))
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>(), Group("Input/output"))
  WorkflowProperty(Labels, Input("L"), Group("Input/output"))
  WorkflowProperty(Outputs, Output("Out"), Group("Input/output"))

EndPropertyDefinitions

Predict::Predict() : _MaximumLayer(-1), _CalculateDeltas(false), _SensitivityRatio(0.02), _SubRegionCount(tbblas::seq<host_tensor_t::dimCount>(1)) {
  setLabel("Predict");
}

PredictChecker predictChecker;

} /* namespace encoder */

} /* namespace gml */
