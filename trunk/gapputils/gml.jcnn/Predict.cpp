/*
 * Predict.cpp
 *
 *  Created on: Dec 02, 2014
 *      Author: tombr
 */

#include "Predict.h"

namespace gml {

namespace jcnn {

BeginPropertyDefinitions(Predict)

  ReflectableBase(DefaultWorkflowElement<Predict>)

  WorkflowProperty(Model, Input("JCNN"), NotNull<Type>())
  WorkflowProperty(LeftInputs, Input("LD"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(RightInputs, Input("RD"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Predict::Predict() {
  setLabel("Predict");
}

PredictChecker predictChecker;

} /* namespace jcnn */

} /* namespace gml */
