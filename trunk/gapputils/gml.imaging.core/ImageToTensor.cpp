/*
 * ImageToTensor.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#include "ImageToTensor.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(ImageToTensor)

  ReflectableBase(DefaultWorkflowElement<ImageToTensor>)

  WorkflowProperty(Image, Input("I"))
  WorkflowProperty(Images, Input("Is"))
  WorkflowProperty(Tensor, Output("T"))
  WorkflowProperty(Tensors, Output("Ts"))

EndPropertyDefinitions

ImageToTensor::ImageToTensor() {
  setLabel("I2T");
}

void ImageToTensor::update(IProgressMonitor* /*montor*/) const {
  if (getImage()) {
    image_t& image = *getImage();

    const int width = image.getSize()[0],
        height = image.getSize()[1],
        depth = image.getSize()[2];

    boost::shared_ptr<tensor_t> output(new tensor_t(width, height, depth, 1));
    std::copy(image.begin(), image.end(), output->begin());

    newState->setTensor(output);
  }

  if (getImages() && getImages()->size()) {
    boost::shared_ptr<v_tensor_t> outputs(new v_tensor_t());

    for (size_t i = 0; i < getImages()->size(); ++i) {
      image_t& image = *getImages()->at(i);

      const int width = image.getSize()[0],
          height = image.getSize()[1],
          depth = image.getSize()[2];

      boost::shared_ptr<tensor_t> output(new tensor_t(width, height, depth, 1));
      std::copy(image.begin(), image.end(), output->begin());
      outputs->push_back(output);
    }
    newState->setTensors(outputs);
  }
}

} /* namespace core */
} /* namespace imaging */
} /* namespace gml */
