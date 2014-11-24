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
#include <tbblas/zeros.hpp>

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
  tensor<float, 3> img(inputs[0]->getSize()[0], inputs[0]->getSize()[1], inputs[0]->getSize()[2]);
  tensor<float, 3> out;

  // Make size check
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->getCount() != count) {
      dlog(Severity::Warning) << "Image " << (i + 1) << " has a different size than the rest of the images. Aborting!";
      return;
    }
  }

  switch (getFunction()) {
  case AggregatorFunction::Average:
    {
      out = zeros<float>(img.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        std::copy(inputs[i]->begin(), inputs[i]->end(), img.begin());
        out = out + img;
      }
      out = out / inputs.size();
    }
    break;

  case AggregatorFunction::Sum:
    {
      out = zeros<float>(img.size());
      for (size_t i = 0; i < inputs.size(); ++i) {
        std::copy(inputs[i]->begin(), inputs[i]->end(), img.begin());
        out = out + img;
      }
    }
    break;

  case AggregatorFunction::Maximum:
    {
      std::copy(inputs[0]->begin(), inputs[0]->end(), img.begin());
      out = img;
      for (size_t i = 1; i < inputs.size(); ++i) {
        std::copy(inputs[i]->begin(), inputs[i]->end(), img.begin());
        out = max(out, img);
      }
    }
    break;

  case AggregatorFunction::Minimum:
    {
      std::copy(inputs[0]->begin(), inputs[0]->end(), img.begin());
      out = img;
      for (size_t i = 1; i < inputs.size(); ++i) {
        std::copy(inputs[i]->begin(), inputs[i]->end(), img.begin());
        out = min(out, img);
      }
    }
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
