/*
 * StackTensors.cpp
 *
 *  Created on: Jun 24, 2013
 *      Author: tombr
 */

#include "StackTensors.h"

#include <capputils/MergeAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(StackTensors)

  ReflectableBase(DefaultWorkflowElement<StackTensors>)

  WorkflowProperty(InputTensors, Input(""), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(OutputTensor, Output(""))

EndPropertyDefinitions

StackTensors::StackTensors() {
  setLabel("Stack");
}

void StackTensors::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  const unsigned dimCount = tensor_t::dimCount;
  typedef tensor_t::dim_t dim_t;

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getInputTensors();
  dim_t size = tensors[0]->size();

  int channels = 0;

  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->size() == size) {
      channels += tensors[i]->size()[dimCount - 1];
    } else {
      dlog(Severity::Warning) << "Tensors sizes must match. Aborting!";
      return;
    }
  }

  dim_t outSize = size, offset(0);
  outSize[dimCount - 1] = channels;

  boost::shared_ptr<tensor_t> output(new tensor_t(outSize));
  for (size_t i = 0; i < tensors.size(); ++i) {
    (*output)[offset, size] = *tensors[i];
    offset[dimCount - 1] += tensors[i]->size()[dimCount - 1];
  }
  newState->setOutputTensor(output);
}

} /* namespace convrbm4d */

} /* namespace gml */
