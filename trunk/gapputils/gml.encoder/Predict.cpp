/*
 * Predict.cpp
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#include "Predict.h"

namespace gml {

namespace encoder {

BeginPropertyDefinitions(Predict)

  ReflectableBase(DefaultWorkflowElement<Predict>)

  WorkflowProperty(Model, Input("ENN"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(FilterBatchSize)
  WorkflowProperty(SubRegionCount, Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Predict::Predict() : _SubRegionCount(tbblas::seq<host_tensor_t::dimCount>(1)) {
  setLabel("Predict");
}

PredictChecker predictChecker;

} /* namespace encoder */

} /* namespace gml */
