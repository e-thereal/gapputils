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
#include <gapputils/ReadOnlyAttribute.h>

#include <culib/CudaImage.h>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

DefineEnum(CombinerMode)

BeginPropertyDefinitions(ImageCombiner)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage1, Input("Img1"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImage2, Input("Img2"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
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
      input1->getSize().y != input2->getSize().y ||
      input1->getSize().z != input2->getSize().z)
  {
    std::cout << "[Warning] No input image given or dimensions don't match. Abording!" << std::endl;
    return;
  }

  const int count = input1->getSize().x * input1->getSize().y * input1->getSize().z;

  boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(input1->getSize(), input1->getVoxelSize()));

  input1->saveDeviceToWorkingCopy();
  input2->saveDeviceToWorkingCopy();

  float* buffer1 = input1->getWorkingCopy();
  float* buffer2 = input2->getWorkingCopy();
  float* buffer = output->getOriginalImage();

  switch (getMode()) {
  case CombinerMode::Add:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] + buffer2[i];
    break;

  case CombinerMode::Subtract:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] - buffer2[i];
    break;

  case CombinerMode::Multiply:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] * buffer2[i];
    break;

  case CombinerMode::Divide:
    for (int i = 0; i < count; ++i)
      buffer[i] = buffer1[i] / buffer2[i];
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
