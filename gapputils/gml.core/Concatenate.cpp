/*
 * Concatenate.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: tombr
 */

#include "Concatenate.h"

#include <capputils/attributes/MergeAttribute.h>

#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(Concatenate)

  ReflectableBase(DefaultWorkflowElement<Concatenate>)

  WorkflowProperty(Inputs, Input("In"), Merge<Type>(), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Concatenate::Concatenate() {
  setLabel("Concatenate");
}

void Concatenate::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  std::vector<boost::shared_ptr<v_data_t> >& inputs = *getInputs();
  boost::shared_ptr<v_data_t> outputs(new v_data_t());

  const size_t vectorCount = inputs[0]->size();
  if (vectorCount < 1) {
    dlog(Severity::Warning) << "Need to have at least one input vector. Aborting!";
    return;
  }

  size_t cumVectorSize = 0;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->size() != vectorCount) {
      dlog(Severity::Warning) << "All inputs must contain the same number of vectors. Aborting!";
      return;
    }
    cumVectorSize += inputs[i]->at(0)->size();
  }

  for (size_t iVec = 0; iVec < vectorCount; ++iVec) {
    boost::shared_ptr<data_t> output(new data_t(cumVectorSize));
    for (size_t iInputs = 0, pos = 0; iInputs < inputs.size(); ++iInputs) {
      data_t& input = *inputs[iInputs]->at(iVec);
      std::copy(input.begin(), input.end(), output->begin() + pos);
      pos += input.size();
    }
    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace gml */
