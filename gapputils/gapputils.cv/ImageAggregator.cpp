/*
 * ImageAggregator.cpp
 *
 *  Created on: May 18, 2012
 *      Author: tombr
 */

#include "ImageAggregator.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageAggregator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  ReflectableProperty(Function, Observe(PROPERTY_ID))
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

  culib::ICudaImage& input = *getInputImage();

  switch (getFunction()) {
  case AggregatorFunction::Average: {
    boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(dim3(input.getSize().x, input.getSize().y)));
    cuda::average(&input, output.get());
    output->saveDeviceToWorkingCopy();
    output->freeCaches();
    data->setOutputImage(output);
    } break;

  case AggregatorFunction::Sum: {
    boost::shared_ptr<culib::ICudaImage> output(new culib::CudaImage(dim3(input.getSize().x, input.getSize().y)));
    cuda::sum(&input, output.get());
    output->saveDeviceToWorkingCopy();
    output->freeCaches();
    data->setOutputImage(output);
    } break;
  }
}

void ImageAggregator::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
