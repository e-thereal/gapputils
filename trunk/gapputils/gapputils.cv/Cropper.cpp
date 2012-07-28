/*
 * Cropper.cpp
 *
 *  Created on: Jul 14, 2011
 *      Author: tombr
 */

#include "Cropper.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Cropper)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("Img"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(Rectangle, Input("Rect"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))
  DefineProperty(OutputImage, Output("Img"), Volatile(), Hide(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

Cropper::Cropper() : data(0) {
  WfeUpdateTimestamp
  setLabel("Cropper");

  Changed.connect(capputils::EventHandler<Cropper>(this, &Cropper::changedHandler));
}

Cropper::~Cropper() {
  if (data)
    delete data;
}

void Cropper::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Cropper::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Cropper();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage() || !getRectangle())
    return;

  image_t* input = getInputImage().get();
  const int left = getRectangle()->getLeft();
  const int top = getRectangle()->getTop();
  const int rwidth = getRectangle()->getWidth();
  const int rheight = getRectangle()->getHeight();
  const int width = input->getSize()[0];
  const int height = input->getSize()[1];

  boost::shared_ptr<image_t> output(new image_t(rwidth, rheight, 1, input->getPixelSize()));

  float* inputBuffer = input->getData();
  float* outputBuffer = output->getData();

  for (int y = 0; y < rheight && y + top < height; ++y) {
    for (int x = 0; x < rwidth && x + left < width; ++x) {
      outputBuffer[y * rwidth + x] = inputBuffer[(y + top) * width + x + left];
    }
  }
  data->setOutputImage(output);
}

void Cropper::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
