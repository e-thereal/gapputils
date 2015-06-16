/*
 * TensorRange.cpp
 *
 *  Created on: May 21, 2015
 *      Author: tombr
 */

#include "TensorRange.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(TensorRange, Description("This module does not create copies of the tensors. It simple copies the pointers to a new vector."))

  ReflectableBase(DefaultWorkflowElement<TensorRange>)

  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(First)
  WorkflowProperty(Count, Description("A value of -1 indicates maximum number of samples."))
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

TensorRange::TensorRange() : _First(0), _Count(-1) {
  setLabel("Range");
}

void TensorRange::update(IProgressMonitor* monitor) const {

  v_host_tensor_t& inputs = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

  size_t lastTensor = (_Count > 0 ? std::min((int)inputs.size(), _First + _Count) : inputs.size());

  for (size_t iTensor = _First; iTensor < lastTensor; ++iTensor) {
    outputs->push_back(inputs[iTensor]);
  }

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
