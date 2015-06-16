/*
 * Flatten.cpp
 *
 *  Created on: Jun 1, 2015
 *      Author: tombr
 */

#include "Flatten.h"

#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(Flatten)

  ReflectableBase(DefaultWorkflowElement<Flatten>)

  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Flatten::Flatten() {
  setLabel("Flat");
}

void Flatten::update(IProgressMonitor* monitor) const {
  v_data_t& inputs = *getInputs();

  size_t count = 0;
  for (size_t i = 0; i < inputs.size(); ++i)
    count += inputs[i]->size();

  boost::shared_ptr<data_t> outputs(new data_t(count));
  count = 0;

  for (size_t i = 0; i < inputs.size(); ++i) {
    std::copy(inputs[i]->begin(), inputs[i]->end(), outputs->begin() + count);
    count += inputs[i]->size();
  }

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace gml */
