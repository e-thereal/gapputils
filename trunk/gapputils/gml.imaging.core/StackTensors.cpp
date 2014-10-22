/*
 * StackTensors.cpp
 *
 *  Created on: Jun 24, 2013
 *      Author: tombr
 */

#include "StackTensors.h"

#include <capputils/attributes/MergeAttribute.h>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(StackTensors, Description("Stacks multiple tensors into one multi-channel tensor or into a vector of tensors."))

  ReflectableBase(DefaultWorkflowElement<StackTensors>)

  WorkflowProperty(InputTensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>(), Description("Input tensors with possible more than 1 channel."))
  WorkflowProperty(Mode, Enumerator<Type>(), Description("SingleTensor: One output tensor is created; TensorVector: A vector of tensors is created."))
  WorkflowProperty(OutputTensor, Output("T"), Description("Multi-channels output tensor"))
  WorkflowProperty(OutputTensors, Output("Ts"), Description("Vector of tensors"))

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

  switch (getMode()) {
  case StackMode::SingleTensor:
    {
      dim_t outSize = size, offset = tbblas::seq<dimCount>(0);
      outSize[dimCount - 1] = channels;

      boost::shared_ptr<tensor_t> output(new tensor_t(outSize));
      for (size_t i = 0; i < tensors.size(); ++i) {
        (*output)[offset, size] = *tensors[i];
        offset[dimCount - 1] += tensors[i]->size()[dimCount - 1];
      }
      newState->setOutputTensor(output);
    }
    break;

  case StackMode::TensorVector:
    {
      boost::shared_ptr<v_tensor_t> outputs(new v_tensor_t());
      for (size_t i = 0; i < tensors.size(); ++i) {
        outputs->push_back(boost::make_shared<tensor_t>(*tensors[i]));
      }
      newState->setOutputTensors(outputs);
    }
    break;
  }
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
