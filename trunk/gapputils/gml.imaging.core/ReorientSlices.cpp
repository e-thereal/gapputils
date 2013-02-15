/*
 * ReorientSlices.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#include "ReorientSlices.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(ReorientSlices)

  ReflectableBase(DefaultWorkflowElement<ReorientSlices>)

  WorkflowProperty(InputImage, Input("In"), NotNull<Type>())
  WorkflowProperty(Orientation, Enumerator<Type>())
  WorkflowProperty(Channels, Description("Number of channels. (1 for grey scale, 3 for RGB)"))
  WorkflowProperty(OutputImage, Output("Out"))

EndPropertyDefinitions

ReorientSlices::ReorientSlices() : _Channels(1) {
  setLabel("Reorient");
}

void ReorientSlices::update(IProgressMonitor* monitor) const {
  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output;

  const int channels = getChannels();

  switch (getOrientation()) {
  case SliceOrientation::Axial:
    output = boost::make_shared<image_t>(input.getSize(), input.getPixelSize());
    std::copy(input.begin(), input.end(), output->begin());
    break;

  case SliceOrientation::Sagital:
    output = boost::make_shared<image_t>(input.getSize()[1], input.getSize()[2] / channels, input.getSize()[0] * channels,
        input.getPixelSize()[1], input.getPixelSize()[2], input.getPixelSize()[0]);

    for (size_t z = 0; z < input.getSize()[2] / channels; ++z) {
      for (size_t y = 0; y < input.getSize()[1]; ++y) {
        for (size_t x = 0; x < input.getSize()[0]; ++x) {
          for (int c = 0; c < channels; ++c) {
            output->getData()[y + output->getSize()[0] * (input.getSize()[2] / channels - z - 1 + output->getSize()[1] * (channels * x + c))] =
                input.getData()[x + input.getSize()[0] * (y + input.getSize()[1] * (channels * z + c))];
          }
        }
      }
    }
    break;

  case SliceOrientation::Coronal:
    output = boost::make_shared<image_t>(input.getSize()[0], input.getSize()[2], input.getSize()[1],
        input.getPixelSize()[0], input.getPixelSize()[2], input.getPixelSize()[1]);

    for (size_t z = 0, i = 0; z < input.getSize()[2]; ++z)
      for (size_t y = 0; y < input.getSize()[1]; ++y)
        for (size_t x = 0; x < input.getSize()[0]; ++x, ++i)
          output->getData()[x + output->getSize()[0] * (input.getSize()[2] - z - 1 + output->getSize()[1] * y)] = input.getData()[i];

    break;
  }

  newState->setOutputImage(output);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
