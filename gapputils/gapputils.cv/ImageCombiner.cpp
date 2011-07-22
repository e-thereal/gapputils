/*
 * ImageCombiner.cpp
 *
 *  Created on: Jul 22, 2011
 *      Author: tombr
 */

#include "ImageCombiner.h"

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

#include <culib/CudaImage.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

DefineEnum(CombinerMode)

BeginPropertyDefinitions(ImageCombiner)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage1, Input("Img1"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImage2, Input("Img2"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), Hide(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  ReflectableProperty(Mode, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ImageCombiner::ImageCombiner() : data(0) {
  WfeUpdateTimestamp
  setLabel("ImageCombiner");

  Changed.connect(capputils::EventHandler<ImageCombiner>(this, &ImageCombiner::changedHandler));
}

ImageCombiner::~ImageCombiner() {
  if (data)
    delete data;
}

void ImageCombiner::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageCombiner::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageCombiner();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<culib::ICudaImage> input1 = getInputImage1();
  boost::shared_ptr<culib::ICudaImage> input2 = getInputImage2();

  if (!input1 || !input2 || input1->getSize().x != input2->getSize().x ||
      input1->getSize().y != input2->getSize().y)
  {
    return;
  }

  const int width = input1->getSize().x;
  const int height = input1->getSize().y;

  boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(dim3(width, height)));

  input1->saveDeviceToWorkingCopy();
  input2->saveDeviceToWorkingCopy();

  float* buffer1 = input1->getWorkingCopy();
  float* buffer2 = input2->getWorkingCopy();
  float* buffer = output->getOriginalImage();

  switch (getMode()) {
  case CombinerMode::Subtract:
    for (int y = 0, i = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++i) {
        buffer[i] = buffer1[i] - buffer2[i];
      }
    }
    break;
  }

  output->resetWorkingCopy();
  data->setOutputImage(output);
}

void ImageCombiner::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
