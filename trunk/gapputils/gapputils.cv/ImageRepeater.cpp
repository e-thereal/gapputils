/*
 * ImageRepeater.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: tombr
 */

#include "ImageRepeater.h"

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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <algorithm>

#include <capputils/Logbook.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageRepeater)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("Img"), ReadOnly(), Volatile(), Observe(PROPERTY_ID))
  DefineProperty(Count, Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), ReadOnly(), Volatile(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ImageRepeater::ImageRepeater() : _Count(1), data(0) {
  WfeUpdateTimestamp
  setLabel("ImageRepeater");

  Changed.connect(capputils::EventHandler<ImageRepeater>(this, &ImageRepeater::changedHandler));
}

ImageRepeater::~ImageRepeater() {
  if (data)
    delete data;
}

void ImageRepeater::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageRepeater::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageRepeater();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage()) {
    getLogbook()(capputils::Severity::Warning) << "No input image given.";
    return;
  }
  image_t& input = *getInputImage();
  const unsigned width = input.getSize()[0], height = input.getSize()[1], depth = input.getSize()[2];
  const unsigned voxelCount = width * height * depth;
  const int count = getCount();
  boost::shared_ptr<image_t> output(
      new image_t(width, height, count * depth, input.getPixelSize()));

  float* image = output->getData();
  for (int i = 0; i < count; ++i, image += voxelCount) {
    std::copy(input.getData(), input.getData() + voxelCount, image);
  }

  data->setOutputImage(output);
}

void ImageRepeater::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
