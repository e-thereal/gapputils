/*
 * Conditional.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: tombr
 */

#include "Conditional.h"

namespace gml {

namespace rbm {

BeginPropertyDefinitions(Conditional)

  ReflectableBase(DefaultWorkflowElement<Conditional>)

  WorkflowProperty(Model, Input("RBM"), NotNull<Type>())
  WorkflowProperty(Given, Input("G"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(FirstGiven, Description("Index of first given unit."))
  WorkflowProperty(IterationCount)

  WorkflowProperty(Inferred, Output("I"))

EndPropertyDefinitions

Conditional::Conditional() : _FirstGiven(0), _IterationCount(10) {
  setLabel("Infer");
}

ConditionalChecker conditionalChecker;

} /* namespace rbm */

} /* namespace gml */
