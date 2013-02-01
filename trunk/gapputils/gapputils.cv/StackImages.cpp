/*
 * StackImages.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#include "StackImages.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <algorithm>
#include <iostream>

#include <capputils/Logbook.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(StackImages)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImages, Input("Imgs"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(InputImage1, Input("I1"), ReadOnly(), Volatile(), Observe(Id))
  DefineProperty(InputImage2, Input("I2"), ReadOnly(), Volatile(), Observe(Id))
  DefineProperty(InputImage3, Input("I3"), ReadOnly(), Volatile(), Observe(Id))
  DefineProperty(InputImage4, Input("I4"), ReadOnly(), Volatile(), Observe(Id))
  DefineProperty(OutputImage, Output("Img"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

StackImages::StackImages() : data(0) {
  WfeUpdateTimestamp
  setLabel("StackImages");

  Changed.connect(capputils::EventHandler<StackImages>(this, &StackImages::changedHandler));
}

StackImages::~StackImages() {
  if (data)
    delete data;
}

void StackImages::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void StackImages::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace capputils;
  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Warning);

  if (!data)
    data = new StackImages();

  if (!capputils::Verifier::Valid(*this)) {
    return;
  }

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

  if (getInputImage3()) {
    image_t& image = *getInputImage3();
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

  if (getInputImage4()) {
    image_t& image = *getInputImage4();
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

  if (getInputImage3()) {
    const image_t& image = *getInputImage3();
    const unsigned count = image.getSize()[0] * image.getSize()[1] * image.getSize()[2];
    std::copy(image.getData(), image.getData() + count, imageData);
    imageData += count;
  }

  if (getInputImage4()) {
    const image_t& image = *getInputImage4();
    const unsigned count = image.getSize()[0] * image.getSize()[1] * image.getSize()[2];
    std::copy(image.getData(), image.getData() + count, imageData);
    imageData += count;
  }

  data->setOutputImage(output);
}

void StackImages::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
