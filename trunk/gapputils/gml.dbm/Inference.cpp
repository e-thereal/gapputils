/*
 * Inference.cpp
 *
 *  Created on: Jul 12, 2013
 *      Author: tombr
 */

#include "Inference.h"

namespace gml {

namespace dbm {

BeginPropertyDefinitions(Inference)

  ReflectableBase(DefaultWorkflowElement<Inference>)

  WorkflowProperty(Model, Input("DBM"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mode, Enumerator<Type>())
  WorkflowProperty(ObservedLayer, Description("Layer of the observed variables. The visible layer has index 0. Hidden layers are indexed starting from 1."))
  WorkflowProperty(QueryLayer, Description("Layer of the query variables. The visible layer has index 0. Hidden layers are indexed starting from 1."))
  WorkflowProperty(Iterations)
  WorkflowProperty(GpuCount)
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

Inference::Inference() : _ObservedLayer(-1), _QueryLayer(-1), _Iterations(1), _GpuCount(1) {
  setLabel("Inference");
}

InferenceChecker inferenceChecker;

} /* namespace dbm */

} /* namespace gml */
