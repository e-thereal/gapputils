/*
 * ImageAggregator.cpp
 *
 *  Created on: May 18, 2012
 *      Author: tombr
 */

#include "ImageAggregator.h"

#include <algorithm>

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ImageAggregator)

  ReflectableBase(DefaultWorkflowElement<ImageAggregator>)
  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(Function, Enumerator<Type>())
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

ImageAggregator::ImageAggregator() {
  setLabel("ImageAggregator");
}

void ImageAggregator::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  image_t& input = *getInputImage();

  boost::shared_ptr<image_t> output(new image_t(input.getSize()[0], input.getSize()[1], 1, input.getPixelSize()));
  tensor<float, 3> img(input.getSize()[0], input.getSize()[1], input.getSize()[2]);
  tensor<float, 3> out(output->getSize()[0], output->getSize()[1], output->getSize()[2]);
  std::copy(input.begin(), input.end(), img.begin());

  switch (getFunction()) {
  case AggregatorFunction::Average:
    out = sum(img, 2);
    out = out / img.size()[2];
    break;

  case AggregatorFunction::Sum:
    out = sum(img, 2);
    break;

  default:
    dlog(Severity::Warning) << "Unsupported aggregation function '" << getFunction() << "'. Aborting!";
  }
  assert(out.count() == output->getCount());
  std::copy(out.begin(), out.end(), output->begin());

  newState->setOutputImage(output);
}

}

}
