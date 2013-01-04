/*
 * Tensor4dToImage.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#include "Tensor4dToImage.h"

#include <algorithm>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Tensor4dToImage)

  ReflectableBase(DefaultWorkflowElement<Tensor4dToImage>)

  WorkflowProperty(Tensor, Input("T"), NotNull<Type>())
  WorkflowProperty(Image, Output("I"))

EndPropertyDefinitions

Tensor4dToImage::Tensor4dToImage() {
  setLabel("T2I");
}

void Tensor4dToImage::update(IProgressMonitor* /*montor*/) const {
  tensor_t& tensor = *getTensor();

  const int width = tensor.size()[0],
      height = tensor.size()[1],
      depth = tensor.size()[2] * tensor.size()[3];

  boost::shared_ptr<image_t> image(new image_t(width, height, depth));
  std::copy(tensor.begin(), tensor.end(), image->begin());

  newState->setImage(image);
}

} /* namespace convrbm4d */
} /* namespace gml */
