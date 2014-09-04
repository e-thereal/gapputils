/*
 * VisualizeWeights.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#include "VisualizeWeights.h"

namespace gml {

namespace dbn {

BeginPropertyDefinitions(VisualizeWeights)

  ReflectableBase(DefaultWorkflowElement<VisualizeWeights>)

  WorkflowProperty(Model, Input("Dbn"), NotNull<Type>())
  WorkflowProperty(FilterBatchLength)
  WorkflowProperty(Weights, Output("W"))

EndPropertyDefinitions

VisualizeWeights::VisualizeWeights() {
  setLabel("Weights");
}

VisualizeWeightsChecker visualizeWeights;

} /* namespace dbn */

} /* namespace gml */
