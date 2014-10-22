/*
 * TransposeTensorsAndChannels.cpp
 *
 *  Created on: Apr 25, 2013
 *      Author: tombr
 */

#include "TransposeTensorsAndChannels.h"

#include <capputils/attributes/DeprecatedAttribute.h>
#include <capputils/attributes/RenamedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(TransposeTensorsAndChannels, Renamed("gml::imaging::core::TransposeTensorsAndChannels"), Deprecated("Use gml::imaging::core::TransposeTensorsAndChannels instead."))

  ReflectableBase(DefaultWorkflowElement<TransposeTensorsAndChannels>)

  WorkflowProperty(InputTensors, Input("Ts"))
  WorkflowProperty(InputTensor, Input("T"))
  WorkflowProperty(OutputTensors, Output("Ts"))
  WorkflowProperty(OutputTensor, Output("T"))

EndPropertyDefinitions

TransposeTensorsAndChannels::TransposeTensorsAndChannels() {
  setLabel("Transpose");
}

void TransposeTensorsAndChannels::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef tensor_t::dim_t dim_t;
  const unsigned dimCount = tensor_t::dimCount;

  std::vector<boost::shared_ptr<tensor_t> > inputs;

  if (getInputTensors()) {
    for (size_t i = 0; i < getInputTensors()->size(); ++i)
      inputs.push_back(getInputTensors()->at(i));
  }

  if (getInputTensor())
    inputs.push_back(getInputTensor());

  if (inputs.size() == 0) {
    dlog(Severity::Warning) << "Insufficient number of input tensors. Aborting!";
    return;
  }

  dim_t inSize = inputs[0]->size();
  dim_t outSize = inSize;
  outSize[dimCount - 1] = inputs.size();
  dim_t layerSize = inSize;
  layerSize[dimCount - 1] = 1;

  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs(
      new std::vector<boost::shared_ptr<tensor_t> >());

  for (int iChannel = 0; iChannel < inSize[dimCount - 1]; ++iChannel) {
    boost::shared_ptr<tensor_t> output(new tensor_t(outSize));

    for (size_t iTensor = 0; iTensor < inputs.size(); ++iTensor)
      (*output)[seq(0,0,0,(int)iTensor), layerSize] = (*inputs[iTensor])[seq(0,0,0,iChannel), layerSize];

    outputs->push_back(output);
  }

  newState->setOutputTensors(outputs);
  if (outputs->size())
    newState->setOutputTensor(outputs->at(0));
}

} /* namespace convrbm4d */

} /* namespace gml */
