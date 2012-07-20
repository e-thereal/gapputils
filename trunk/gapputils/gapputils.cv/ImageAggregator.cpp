/*
 * ImageAggregator.cpp
 *
 *  Created on: May 18, 2012
 *      Author: tombr
 */

#include "ImageAggregator.h"

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

#include <gapputils.cv.cuda/aggregate.h>
#include <culib/CudaImage.h>

#include <algorithm>

#include "util.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageAggregator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(Function, Enumerator<AggregatorFunction>(), Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ImageAggregator::ImageAggregator() : data(0) {
  WfeUpdateTimestamp
  setLabel("ImageAggregator");

  Changed.connect(capputils::EventHandler<ImageAggregator>(this, &ImageAggregator::changedHandler));
}

ImageAggregator::~ImageAggregator() {
  if (data)
    delete data;
}

void ImageAggregator::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageAggregator::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageAggregator();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInputImage())
    return;

  boost::shared_ptr<culib::ICudaImage> input = make_cuda_image(*getInputImage());
  boost::shared_ptr<image_t> output(new image_t(getInputImage()->getSize(), getInputImage()->getPixelSize()));
  unsigned count = input->getSize().x * input->getSize().y * input->getSize().z;

  switch (getFunction()) {
  case AggregatorFunction::Average: {
    cuda::average(input.get(), input.get());
    input->saveDeviceToWorkingCopy();
    std::copy(input->getWorkingCopy(), input->getWorkingCopy() + count, output->getData());
    } break;

  case AggregatorFunction::Sum: {
    cuda::sum(input.get(), input.get());
    input->saveDeviceToWorkingCopy();
    std::copy(input->getWorkingCopy(), input->getWorkingCopy() + count, output->getData());
    } break;
  }

  data->setOutputImage(output);
}

void ImageAggregator::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
