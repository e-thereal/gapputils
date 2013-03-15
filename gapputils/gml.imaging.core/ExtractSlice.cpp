/*
 * ExtractSlice.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: tombr
 */

#include "ExtractSlice.h"

#include <algorithm>
#include <cmath>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(ExtractSlice)

  ReflectableBase(DefaultWorkflowElement<ExtractSlice>)

  WorkflowProperty(Volume, Input(""), NotNull<Type>())
  WorkflowProperty(SliceIndex)
  WorkflowProperty(Channels, Description("Number of channels. (1 for grey scale, 3 for RGB)"))
  WorkflowProperty(Slice, Output(""))

EndPropertyDefinitions

ExtractSlice::ExtractSlice() {
  setLabel("Slice");
}

void ExtractSlice::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  image_t& input = *getVolume();

  if (getChannels() > (int)input.getSize()[2]) {
    dlog(Severity::Warning) << "Channels must not be greater than the total number of slices of the input volume. Aborting!";
    return;
  }

  if (getSliceIndex() < 0 || (getSliceIndex() + 1) * getChannels() > (int)input.getSize()[2]) {
    dlog(Severity::Warning) << "Not a valid slice index. Aborting!";
    return;
  }

  boost::shared_ptr<image_t> output(new image_t(input.getSize()[0], input.getSize()[1], getChannels(), input.getPixelSize()));
  std::copy(
      input.getData() + getSliceIndex() * output->getCount(),
      input.getData() + (getSliceIndex() + 1) * output->getCount(),
      output->getData()
  );

  newState->setSlice(output);
}

} /* namespace core */
} /* namespace imaging */
} /* namespace gml */
