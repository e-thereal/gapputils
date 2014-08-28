/*
 * BackpropWeights.cpp
 *
 *  Created on: 2014-08-16
 *      Author: tombr
 */

#include "BackpropWeights.h"

#include <capputils/attributes/GreaterThanAttribute.h>

namespace gml {

namespace cnn {

BeginPropertyDefinitions(BackpropWeights)

  ReflectableBase(DefaultWorkflowElement<BackpropWeights>)

  WorkflowProperty(Model, Input("CNN"), NotNull<Type>())
  WorkflowProperty(Layer, Description("The dense layer from which the backpropagation of weights starts."), GreaterThan<Type>(-1))
  WorkflowProperty(Weights, Output("W"), Description("Backpropagated weights"))

EndPropertyDefinitions

BackpropWeights::BackpropWeights() : _Layer(0) {
  setLabel("Weights");
}

BackpropWeightsChecker backpropWeightsChecker;

} /* namespace cnn */

} /* namespace gml */
