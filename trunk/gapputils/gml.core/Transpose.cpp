/*
 * Transpose.cpp
 *
 *  Created on: Apr 23, 2015
 *      Author: tombr
 */

#include "Transpose.h"

namespace gml {

namespace core {

BeginPropertyDefinitions(Transpose)

  ReflectableBase(DefaultWorkflowElement<Transpose>)

  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Transpose::Transpose() {
  setLabel("Trans");
}

void Transpose::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  v_data_t& inputs = *getInputs();
  boost::shared_ptr<v_data_t> outputs(new v_data_t());

  size_t count = inputs.size();
  size_t size = inputs[0]->size();

  for (size_t i = 0; i < count; ++i) {
    if (inputs[i]->size() != size) {
      dlog(Severity::Warning) << "All input vectors must of the same size. Aborting!";
      return;
    }
  }

  for (size_t iPos = 0; iPos < size; ++iPos) {
    boost::shared_ptr<data_t> output(new data_t(count));
    for (size_t iVec = 0; iVec < count; ++iVec) {
      output->at(iVec) = inputs[iVec]->at(iPos);
    }
    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace gml */
