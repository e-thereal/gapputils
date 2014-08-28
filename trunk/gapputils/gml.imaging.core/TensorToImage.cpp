/*
 * TensorToImage.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#include "TensorToImage.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(TensorToImage)

  ReflectableBase(DefaultWorkflowElement<TensorToImage>)

  WorkflowProperty(Tensor, Input("T"))
  WorkflowProperty(Tensors, Input("Ts"))
  WorkflowProperty(Spacing, Description("Voxel spacing in mm."))
  WorkflowProperty(Image, Output("I"))
  WorkflowProperty(Images, Output("Is"))

EndPropertyDefinitions

TensorToImage::TensorToImage() {
  setLabel("T2I");
  std::vector<double> spacing(3);
  spacing[0] = 1;
  spacing[1] = 1;
  spacing[2] = 1;
  setSpacing(spacing);
}

void TensorToImage::update(IProgressMonitor* /*montor*/) const {
  Logbook& dlog = getLogbook();

  if (getSpacing().size() != 3) {
    dlog(Severity::Warning) << "Spacing must have exactly 3 values. Aborting!";
    return;
  }

  if (getTensor()) {
    tensor_t& tensor = *getTensor();

    const int width = tensor.size()[0],
        height = tensor.size()[1],
        depth = tensor.size()[2] * tensor.size()[3];

    boost::shared_ptr<image_t> image(new image_t(width, height, depth, getSpacing()[0] * 1000, getSpacing()[1] * 1000, getSpacing()[2] * 1000));
    std::copy(tensor.begin(), tensor.end(), image->begin());

    newState->setImage(image);
  }

  if (getTensors()) {
    boost::shared_ptr<v_image_t> images(new v_image_t);

    for (size_t i = 0; i < getTensors()->size(); ++i) {
      tensor_t& tensor = *getTensors()->at(i);

      const int width = tensor.size()[0],
          height = tensor.size()[1],
          depth = tensor.size()[2] * tensor.size()[3];

      boost::shared_ptr<image_t> image(new image_t(width, height, depth, getSpacing()[0] * 1000, getSpacing()[1] * 1000, getSpacing()[2] * 1000));
      std::copy(tensor.begin(), tensor.end(), image->begin());
      images->push_back(image);
    }

    newState->setImages(images);
  }
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
