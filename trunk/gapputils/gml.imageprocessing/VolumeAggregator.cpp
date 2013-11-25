/*
 * VolumeAggregator.cpp
 *
 *  Created on: Nov 19, 2013
 *      Author: tombr
 */

#include "VolumeAggregator.h"

#include <algorithm>

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(VolumeAggregator)

  ReflectableBase(DefaultWorkflowElement<VolumeAggregator>)

  WorkflowProperty(InputImages, Input(""), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Function, Enumerator<Type>())
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

VolumeAggregator::VolumeAggregator() {
  setLabel("VolumeAggregagor");
}

void VolumeAggregator::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  std::vector<boost::shared_ptr<image_t> >& inputs = *getInputImages();
  size_t count = inputs[0]->getCount();

  boost::shared_ptr<image_t> output(new image_t(inputs[0]->getSize(), inputs[0]->getPixelSize()));
  tensor<float, 4> img(inputs[0]->getSize()[0], inputs[0]->getSize()[1], inputs[0]->getSize()[2], inputs.size());
  tensor<float, 4> out(output->getSize()[0], output->getSize()[1], output->getSize()[2], 1);

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->getCount() != count) {
      dlog(Severity::Warning) << "Image " << (i + 1) << " has a different size than the rest of the images. Aborting!";
      return;
    }
    assert(img.end() >= img.begin() + i * inputs[i]->getCount() + (inputs[i]->end() - inputs[i]->begin()));
    std::copy(inputs[i]->begin(), inputs[i]->end(), img.begin() + i * inputs[i]->getCount());
  }

  switch (getFunction()) {
  case AggregatorFunction::Average:
    out = sum(img, 3);
    out = out / inputs.size();
    break;

  case AggregatorFunction::Sum:
    out = sum(img, 3);
    break;

  default:
    dlog(Severity::Warning) << "Unsupported aggregation function '" << getFunction() << "'. Aborting!";
  }

  assert(out.count() == output->getCount());
  std::copy(out.begin(), out.end(), output->begin());

  newState->setOutputImage(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
