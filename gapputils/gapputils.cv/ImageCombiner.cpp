/*
 * ImageCombiner.cpp
 *
 *  Created on: Jul 22, 2011
 *      Author: tombr
 */

#include "ImageCombiner.h"

#include <capputils/EnumeratorAttribute.h>
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

#include <iostream>

#include <capputils/Logbook.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageCombiner)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage1, Input("Img1"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(InputImage2, Input("Img2"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Mode, Enumerator<CombinerMode>(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

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

  boost::shared_ptr<image_t> input1 = getInputImage1();
  boost::shared_ptr<image_t> input2 = getInputImage2();

  if (!input1 || !input2 || input1->getSize()[0] != input2->getSize()[0] ||
      input1->getSize()[1] != input2->getSize()[1] ||
      input1->getSize()[2] != input2->getSize()[2])
  {
    getLogbook()(capputils::Severity::Warning) << "No input image given or dimensions don't match. Abording!";
    return;
  }

  const int count = input1->getSize()[0] * input1->getSize()[1] * input1->getSize()[2];

  boost::shared_ptr<image_t> output(new image_t(input1->getSize(), input1->getPixelSize()));

  float* buffer1 = input1->getData();
  float* buffer2 = input2->getData();
  float* buffer = output->getData();

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

  data->setOutputImage(output);
}

void ImageCombiner::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
