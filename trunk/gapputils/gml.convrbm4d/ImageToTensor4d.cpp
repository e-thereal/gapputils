/*
 * ImageToTensor4d.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#include "ImageToTensor4d.h"

#include <algorithm>

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(ImageToTensor4d)

  ReflectableBase(DefaultWorkflowElement<ImageToTensor4d>)

  WorkflowProperty(Image, Input("I"), NotNull<Type>())
  WorkflowProperty(Tensor, Output("T"))

EndPropertyDefinitions

ImageToTensor4d::ImageToTensor4d() {
  setLabel("I2T");
}

void ImageToTensor4d::update(IProgressMonitor* /*montor*/) const {
  image_t& image = *getImage();

  const int width = image.getSize()[0],
      height = image.getSize()[1],
      depth = image.getSize()[2];

  boost::shared_ptr<tensor_t> output(new tensor_t(width, height, depth, 1));
  std::copy(image.begin(), image.end(), output->begin());

  newState->setTensor(output);
}

} /* namespace convrbm4d */
} /* namespace gml */
