/*
 * Normalize.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: tombr
 */

#include "Normalize.h"

namespace gml {

namespace core {

BeginPropertyDefinitions(Normalize)

  ReflectableBase(DefaultWorkflowElement<Normalize>)

  WorkflowProperty(Inputs, Input(""), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mean)
  WorkflowProperty(Stddev)
  WorkflowProperty(Outputs, Output(""))

EndPropertyDefinitions

Normalize::Normalize() {
  setLabel("Norm");
}

void Normalize::update(IProgressMonitor* monitor) const {
  data_t& inputs = *getInputs();
  boost::shared_ptr<data_t> outputs(new data_t(inputs.size()));

  const value_t mean = getMean(), sd = getStddev();

  for (size_t i = 0; i < inputs.size(); ++i)
    outputs->at(i) = (inputs[i] - mean) / sd;

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace gml */
