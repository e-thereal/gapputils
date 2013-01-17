/*
 * Encoder.cpp
 *
 *  Created on: Jan 15, 2013
 *      Author: tombr
 */

#include "Encoder.h"

namespace gml {

namespace rbm {

BeginPropertyDefinitions(Encoder)

  ReflectableBase(DefaultWorkflowElement<Encoder>)

  WorkflowProperty(Model, Input("RBM"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Encoder::Encoder() {
  setLabel("Encoder");
}

EncoderChecker encoderChecker;

}

}