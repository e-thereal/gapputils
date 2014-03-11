/*
 * Stack.cpp
 *
 *  Created on: Jan 22, 2014
 *      Author: tombr
 */

#include "Stack.h"

#include <capputils/attributes/MergeAttribute.h>

namespace gml {

namespace core {

BeginPropertyDefinitions(Stack)

  ReflectableBase(DefaultWorkflowElement<Stack>)

  WorkflowProperty(Inputs, Input("V"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(Outputs, Output("V"))

EndPropertyDefinitions

Stack::Stack() {
  setLabel("Stack");
}

void Stack::update(IProgressMonitor* monitor) const {

  vv_data_t& inputs = *getInputs();
  boost::shared_ptr<v_data_t> outputs(new v_data_t());

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!inputs[i])
      continue;
    for (size_t j = 0; j < inputs[i]->size(); ++j) {
      if (!inputs[i]->at(j))
        continue;
      outputs->push_back(boost::make_shared<data_t>(inputs[i]->at(j)->begin(), inputs[i]->at(j)->end()));
    }
  }

  newState->setOutputs(outputs);
}

} /* namespace core */
} /* namespace gml */
