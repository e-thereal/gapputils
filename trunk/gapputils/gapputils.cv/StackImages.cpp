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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <algorithm>
#include <iostream>

#include <culib/CudaImage.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(StackImages)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImages, Input("Imgs"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputImage, Output("Img"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

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
  if (!data)
    data = new StackImages();

  if (!capputils::Verifier::Valid(*this) || !getInputImages() || !getInputImages()->size()
      || !getInputImages()->at(0))
  {
    return;
  }

  std::vector<boost::shared_ptr<culib::ICudaImage> >& inputs = *getInputImages();
  const int width = inputs[0]->getSize().x;
  const int height = inputs[0]->getSize().y;

  int depth = 0;
  for (int i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->getSize().x != width || inputs[i]->getSize().y != height) {
      std::cout << "[Warning] Size mismatch. Aborting." << std::endl;
      return;
    }
    depth += inputs[i]->getSize().z;
  }

  boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(dim3(width, height, depth)));
  float* imageData = output->getOriginalImage();
  for (int i = 0; i < inputs.size(); ++i) {
    const culib::ICudaImage& image = *inputs[i];

    const int count = image.getSize().x * image.getSize().y * image.getSize().z;
    std::copy(image.getWorkingCopy(), image.getWorkingCopy() + count, imageData);
    imageData += count;
  }
  output->resetWorkingCopy();

  data->setOutputImage(output);
}

void StackImages::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
