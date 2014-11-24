/*
 * ManipulateTensor.cpp
 *
 *  Created on: Nov 10, 2014
 *      Author: tombr
 */

#include "ManipulateTensor.h"

#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ManipulateTensor)

  ReflectableBase(DefaultWorkflowElement<ManipulateTensor>)

  WorkflowProperty(Input, Input("In"), NotNull<Type>())
  WorkflowProperty(Mask)
  WorkflowProperty(Output, Output("Out"))

EndPropertyDefinitions

ManipulateTensor::ManipulateTensor() {
  setLabel("Mask");
}

void ManipulateTensor::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  host_tensor_t& input = *getInput();
  boost::shared_ptr<host_tensor_t> output(new host_tensor_t(input));

  host_tensor_t::dim_t channelSize = input.size();
  channelSize[3] = 1;

  const std::vector<int>& mask = getMask();

  for (int i = 0; i < input.size()[3]; ++i) {
    if (i >= (int)mask.size() || !mask[i])
      (*output)[seq(0,0,0,i), channelSize] = zeros<float>(channelSize);
//    if (i < (int)mask.size())
//      (*output)[seq(0,0,0,i), channelSize] = mask[i] * ones<float>(channelSize);
  }

  newState->setOutput(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
