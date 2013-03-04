/*
 * NormalizeContrast.cpp
 *
 *  Created on: Mar 1, 2013
 *      Author: tombr
 */

#include "NormalizeContrast.h"

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

#include <algorithm>

namespace gml {
namespace imageprocessing {

BeginPropertyDefinitions(NormalizeContrast)

  ReflectableBase(DefaultWorkflowElement<NormalizeContrast>)

  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

NormalizeContrast::NormalizeContrast() {
  setLabel("Norm");
}

void NormalizeContrast::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

  tensor<float, 3> image1(input.getSize()[0], input.getSize()[1], input.getSize()[2]), image2;
  std::copy(input.begin(), input.end(), image1.begin());

  image2 = max(0, image1) / max(image1);
  std::copy(image2.begin(), image2.end(), output->begin());

  newState->setOutputImage(output);
}

} /* namespace imageprocessing */
} /* namespace gml */
