/*
 * MaskChannels.cpp
 *
 *  Created on: 2013-05-20
 *      Author: tombr
 */

#include "MaskChannels.h"

#include <tbblas/zeros.hpp>

#include <capputils/attributes/DeprecatedAttribute.h>
#include <capputils/attributes/RenamedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(MaskChannels, Renamed("gml::imageprocessing::MaskChannels"), Deprecated("Use gml::imageprocessing::MaskChannels instead."))

  ReflectableBase(DefaultWorkflowElement<MaskChannels>)

  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(ChannelMask, Input("Mask"), NotNull<Type>())
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

MaskChannels::MaskChannels() {
  setLabel("MaskChannels");
}

void MaskChannels::update(IProgressMonitor* monitor) const {
  // Make sure that ChannelMask has the same size as the tensor number of channels.
  // Black out entire channels from the 4D tensors

  using namespace tbblas;

  typedef tensor_t::dim_t dim_t;
  typedef tensor_t::value_t value_t;

  const unsigned dimCount = tensor_t::dimCount;

  std::vector<boost::shared_ptr<tensor_t> >& inputs = *getInputs();
  std::vector<double>& channelMask = *getChannelMask();
  boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > > outputs = boost::make_shared<std::vector<boost::shared_ptr<tensor_t> > >();

  Logbook& dlog = getLogbook();

  for (size_t i = 0; i < inputs.size(); ++i) {
    tensor_t& input = *inputs[i];
    boost::shared_ptr<tensor_t> output = boost::make_shared<tensor_t>(zeros<value_t>(input.size()));

    dim_t sliceSize = input.size();
    const int channelCount = sliceSize[dimCount - 1];
    sliceSize[dimCount - 1] = 1;

    if ((int)channelMask.size() < channelCount) {
      dlog(Severity::Warning) << "Channel count mismatch (" << channelMask.size() << " < " << channelCount << "). Skipping tensor!";
      continue;
    }

    for (int iChannel = 0; iChannel < channelCount; ++iChannel) {
      if (channelMask[iChannel] > 0)
        (*output)[seq(0,0,0,iChannel), sliceSize] = input[seq(0,0,0,iChannel), sliceSize];
    }

    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

} /* namespace convrbm4d */
} /* namespace gml */
