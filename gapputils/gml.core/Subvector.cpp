/*
 * Subvector.cpp
 *
 *  Created on: Jan 29, 2014
 *      Author: tombr
 */

#include "Subvector.h"

#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(Subvector)

  ReflectableBase(DefaultWorkflowElement<Subvector>)

  WorkflowProperty(Inputs, Input(""), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(StartIndex)
  WorkflowProperty(Length, Description("A length of -1 indicates reading until the end."))
  WorkflowProperty(Outputs, Output(""))

EndPropertyDefinitions

Subvector::Subvector() : _StartIndex(0), _Length(-1) {
  setLabel("Sub");
}

void Subvector::update(IProgressMonitor* monitor) const {
  v_data_t& inputs = *getInputs();
  boost::shared_ptr<v_data_t> outputs(new v_data_t());

  const int start = getStartIndex();
  const int length = getLength();

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (start >= (int)inputs[i]->size())
      continue;
    if (length > 0)
      outputs->push_back(boost::make_shared<data_t>(inputs[i]->begin() + start, std::min(inputs[i]->begin() + start + length, inputs[i]->end())));
    else
      outputs->push_back(boost::make_shared<data_t>(inputs[i]->begin() + start, inputs[i]->end()));
  }

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace gml */
