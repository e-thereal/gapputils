/*
 * Predict.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#include "Predict.h"

namespace gml {

namespace nn {

BeginPropertyDefinitions(Predict)

  ReflectableBase(DefaultWorkflowElement<Predict>)

  WorkflowProperty(Model, Input("NN"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Predict::Predict() {
  setLabel("Predict");
}

PredictChecker predictChecker;

} /* namespace nn */
} /* namespace gml */
