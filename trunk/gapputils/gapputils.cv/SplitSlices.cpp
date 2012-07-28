/*
 * SplitSlices.cpp
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#include "SplitSlices.h"

#include <capputils/NotNullAttribute.h>

#include <algorithm>

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(SplitSlices)
  using namespace capputils::attributes;
  using namespace gapputils::attributes;

  ReflectableBase(workflow::DefaultWorkflowElement<SplitSlices>)

  WorkflowProperty(Volume, Input(""), NotNull<Type>());
  WorkflowProperty(Slices, Output(""));

EndPropertyDefinitions

SplitSlices::SplitSlices() {
  setLabel("Splitter");
}

SplitSlices::~SplitSlices() {
}

void SplitSlices::update(workflow::IProgressMonitor* /*monitor*/) const {
  using namespace capputils;

  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > slices(new std::vector<boost::shared_ptr<image_t> >());
  image_t& volume = *getVolume();
  const unsigned width = volume.getSize()[0], height = volume.getSize()[1], depth = volume.getSize()[2];
  const unsigned pitch = width * height;

  float* volumeData = volume.getData();

  for (unsigned z = 0; z < depth; ++z, volumeData += pitch) {
    boost::shared_ptr<image_t> slice(new image_t(width, height, 1, volume.getPixelSize()));
    std::copy(volumeData, volumeData + pitch, slice->getData());
    slices->push_back(slice);
  }

  newState->setSlices(slices);
}

} /* namespace cv */

} /* namespace gapputils */
