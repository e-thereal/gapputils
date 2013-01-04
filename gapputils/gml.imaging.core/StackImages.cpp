/*
 * StackImages.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#include "StackImages.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(StackImages)
  ReflectableBase(DefaultWorkflowElement<StackImages>)

  WorkflowProperty(InputImages, Input("Imgs"))
  WorkflowProperty(InputImage1, Input("I1"))
  WorkflowProperty(InputImage2, Input("I2"))
  WorkflowProperty(OutputImage, Output("Img"))

EndPropertyDefinitions

StackImages::StackImages() {
  setLabel("Stack");
}

void StackImages::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Warning);

  // Get and check the dimensions
  unsigned width = 0, height = 0, depth = 0;
  if (getInputImages()) {
    std::vector<boost::shared_ptr<image_t> >& inputs = *getInputImages();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (depth == 0) {
        width = inputs[i]->getSize()[0];
        height = inputs[i]->getSize()[1];
      }
      if (inputs[i]->getSize()[0] != width || inputs[i]->getSize()[1] != height) {
        dlog() << "Size mismatch. Aborting.";
        return;
      }
      depth += inputs[i]->getSize()[2];
    }
  }

  if (getInputImage1()) {
    image_t& image = *getInputImage1();
    if (depth == 0) {
      width = image.getSize()[0];
      height = image.getSize()[1];
    }

    if (image.getSize()[0] != width || image.getSize()[1] != height) {
      dlog() << "Size mismatch. Aborting.";
      return;
    }
    depth += image.getSize()[2];
  }

  if (getInputImage2()) {
    image_t& image = *getInputImage2();
    if (depth == 0) {
      width = image.getSize()[0];
      height = image.getSize()[1];
    }

    if (image.getSize()[0] != width || image.getSize()[1] != height) {
      dlog() << "Size mismatch. Aborting.";
      return;
    }
    depth += image.getSize()[2];
  }

  if (depth == 0) {
    dlog() << "No input images found.";
    return;
  }

  boost::shared_ptr<image_t> output(new image_t(width, height, depth));
  float* imageData = output->getData();
  if (getInputImages()) {
    std::vector<boost::shared_ptr<image_t> >& inputs = *getInputImages();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      const image_t& image = *inputs[i];

      const unsigned count = image.getSize()[0] * image.getSize()[1] * image.getSize()[2];
      std::copy(image.getData(), image.getData() + count, imageData);
      imageData += count;
    }
  }

  if (getInputImage1()) {
    const image_t& image = *getInputImage1();
    const unsigned count = image.getSize()[0] * image.getSize()[1] * image.getSize()[2];
    std::copy(image.getData(), image.getData() + count, imageData);
    imageData += count;
  }

  if (getInputImage2()) {
    const image_t& image = *getInputImage2();
    const unsigned count = image.getSize()[0] * image.getSize()[1] * image.getSize()[2];
    std::copy(image.getData(), image.getData() + count, imageData);
    imageData += count;
  }

  newState->setOutputImage(output);
}

}

}

}
