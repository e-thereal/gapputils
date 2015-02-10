/*
 * Predict.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: tombr
 */

#include "Predict.h"

namespace gml {

namespace cnn {

BeginPropertyDefinitions(Predict)

  ReflectableBase(DefaultWorkflowElement<Predict>)

  WorkflowProperty(Model, Input("CNN"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Outputs, Output("Out"))
//  WorkflowProperty(FirstLayer, Output("First"))

EndPropertyDefinitions

Predict::Predict() {
  setLabel("Predict");
}

PredictChecker predictChecker;

} /* namespace cnn */

} /* namespace gml */
