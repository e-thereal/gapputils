/*
 * PadTensors.cpp
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#include "PadTensors.h"

#include <tbblas/zeros.hpp>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(PadTensors)

  ReflectableBase(DefaultWorkflowElement<PadTensors>)

  WorkflowProperty(InputTensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Width)
  WorkflowProperty(Height)
  WorkflowProperty(Depth)
  WorkflowProperty(OutputTensors, Output("Ts"))

EndPropertyDefinitions

PadTensors::PadTensors() : _Width(0), _Height(0), _Depth(0) {
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

  std::vector<boost::shared_ptr<tensor_t> >& inputs = *getInputTensors();
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<tensor_t> >());

  dim_t size = seq(getWidth(), getHeight(), getDepth(), 0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    tensor_t& kern = *inputs[i];
    size[dimCount - 1] = kern.size()[dimCount - 1];
    dim_t topleft = size / 2 - kern.size() / 2;
    boost::shared_ptr<tensor_t> pad(new tensor_t(zeros<value_t>(size)));
    (*pad)[topleft, kern.size()] = kern;
    outputs->push_back(pad);
  }

  newState->setOutputTensors(outputs);
}

} /* namespace convrbm4d */

} /* namespace gml */
