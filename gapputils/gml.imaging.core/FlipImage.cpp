/*
 * FlipImage.cpp
 *
 *  Created on: Nov 28, 2013
 *      Author: tombr
 */

#include "FlipImage.h"

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(FlipImage)

  ReflectableBase(DefaultWorkflowElement<FlipImage>)

  WorkflowProperty(Input, Input(""), NotNull<Type>())
  WorkflowProperty(FlipAxis, Enumerator<Type>())
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

FlipImage::FlipImage() {
  setLabel("Flip");
}

void FlipImage::update(IProgressMonitor* monitor) const {
  image_t& input = *getInput();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

  int width = input.getSize()[0], height = input.getSize()[1], depth = input.getSize()[2];
  image_t::value_t* inbuf = input.getData();
  image_t::value_t* outbuf = output->getData();

  switch (getFlipAxis()) {
  case FlipAxis::LeftRight:
    for (int z = 0, i = 0; z < depth; ++z) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          outbuf[(width - x - 1) + width * (y + z * height)] = inbuf[i];
        }
      }
    }
    break;

  case FlipAxis::AnteriorPosterior:
    for (int z = 0, i = 0; z < depth; ++z) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          outbuf[x + width * ((height - y - 1) + z * height)] = inbuf[i];
        }
      }
    }
    break;

  case FlipAxis::SuperiorInferior:
    for (int z = 0, i = 0; z < depth; ++z) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x, ++i) {
          outbuf[x + width * (y + (depth - z - 1) * height)] = inbuf[i];
        }
      }
    }
    break;
  }

  newState->setOutput(output);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
