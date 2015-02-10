/*
 * PadTensors.cpp
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#include "PadTensors.h"

#include <tbblas/zeros.hpp>
#include <tbblas/io.hpp>

#include <capputils/attributes/OrAttribute.h>
#include <capputils/attributes/IsNullAttribute.h>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(PadTensors)

  ReflectableBase(DefaultWorkflowElement<PadTensors>)

  WorkflowProperty(InputTensor, Input("T"))
  WorkflowProperty(InputTensors, Input("Ts"), Or(IsNull<Type>(), NotEmpty<Type>()))
  WorkflowProperty(Width)
  WorkflowProperty(Height)
  WorkflowProperty(Depth)
  WorkflowProperty(ReversePadding, Flag())
  WorkflowProperty(OutputTensor, Output("T"))
  WorkflowProperty(OutputTensors, Output("Ts"))

EndPropertyDefinitions

PadTensors::PadTensors() : _Width(0), _Height(0), _Depth(0), _ReversePadding(false) {
  setLabel("Padding");
}

void PadTensors::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef tensor_t::dim_t dim_t;
  typedef tensor_t::value_t value_t;
  const int dimCount = tensor_t::dimCount;

  if (getWidth() <= 0 || getHeight() <= 0 || getDepth() <= 0) {
    dlog(Severity::Warning) << "Dimensions must be positive. Aborting!";
    return;
  }

  if (getInputTensor()) {
    if (!getReversePadding()) {
      dim_t padSize = seq(getWidth(), getHeight(), getDepth(), 0);
      tensor_t& kern = *getInputTensor();
      padSize[dimCount - 1] = kern.size()[dimCount - 1];
      if (padSize == kern.size()) {
        newState->setOutputTensor(getInputTensor());
      } else {
        dim_t topleft = padSize / 2 - kern.size() / 2;
        boost::shared_ptr<tensor_t> pad(new tensor_t(zeros<value_t>(padSize)));
        (*pad)[topleft, kern.size()] = kern;
        newState->setOutputTensor(pad);
      }
    } else {
      dim_t kernSize = seq(getWidth(), getHeight(), getDepth(), 0);
      tensor_t& pad = *getInputTensor();
      kernSize[dimCount - 1] = pad.size()[dimCount - 1];
      if (kernSize == pad.size()) {
        newState->setOutputTensor(getInputTensor());
      } else {
        dim_t topleft = pad.size() / 2 - kernSize / 2;
        boost::shared_ptr<tensor_t> kern(new tensor_t(zeros<value_t>(kernSize)));
        *kern = pad[topleft, kernSize];
        newState->setOutputTensor(kern);
      }
    }
  }

  if (getInputTensors()) {
    v_tensor_t& inputs = *getInputTensors();
    boost::shared_ptr<v_tensor_t> outputs(new v_tensor_t());

    if (!getReversePadding()) {
      dim_t padSize = seq(getWidth(), getHeight(), getDepth(), 0);
      for (size_t i = 0; i < inputs.size(); ++i) {
        tensor_t& kern = *inputs[i];
        padSize[dimCount - 1] = kern.size()[dimCount - 1];
        if (padSize == kern.size()) {
          outputs->push_back(inputs[i]);
        } else {
          dim_t topleft = padSize / 2 - kern.size() / 2;
          boost::shared_ptr<tensor_t> pad(new tensor_t(zeros<value_t>(padSize)));
          (*pad)[topleft, kern.size()] = kern;
          outputs->push_back(pad);
        }
      }
    } else {
      dim_t kernSize = seq(getWidth(), getHeight(), getDepth(), 0);
      for (size_t i = 0; i < inputs.size(); ++i) {
        tensor_t& pad = *inputs[i];
        kernSize[dimCount - 1] = pad.size()[dimCount - 1];
        if (pad.size() == kernSize) {
          outputs->push_back(inputs[i]);
        } else {
          dim_t topleft = pad.size() / 2 - kernSize / 2;
          boost::shared_ptr<tensor_t> kern(new tensor_t(zeros<value_t>(kernSize)));
          *kern = pad[topleft, kernSize];
          outputs->push_back(kern);
        }
      }
    }

    newState->setOutputTensors(outputs);
  }
}

} /* namespace imageprocessing */

} /* namespace gml */
