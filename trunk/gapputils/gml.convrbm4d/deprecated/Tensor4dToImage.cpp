/*
 * Tensor4dToImage.cpp
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#include "Tensor4dToImage.h"

#include <algorithm>

#include <capputils/attributes/RenamedAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Tensor4dToImage, Renamed("gml::imaging::core::TensorToImage"), Deprecated("Use gml::imaging::core::TensorToImage instead."))

  ReflectableBase(DefaultWorkflowElement<Tensor4dToImage>)

  WorkflowProperty(Tensor, Input("T"), NotNull<Type>())
  WorkflowProperty(Spacing, Description("Voxel spacing in mm."))
  WorkflowProperty(Image, Output("I"))

EndPropertyDefinitions

Tensor4dToImage::Tensor4dToImage() {
  setLabel("T2I");
  std::vector<double> spacing(3);
  spacing[0] = 1;
  spacing[1] = 1;
  spacing[2] = 1;
  setSpacing(spacing);
}

void Tensor4dToImage::update(IProgressMonitor* /*montor*/) const {
  Logbook& dlog = getLogbook();

  if (getSpacing().size() != 3) {
    dlog(Severity::Warning) << "Spacing must have exactly 3 values. Aborting!";
    return;
  }

  tensor_t& tensor = *getTensor();

  const int width = tensor.size()[0],
      height = tensor.size()[1],
      depth = tensor.size()[2] * tensor.size()[3];

  boost::shared_ptr<image_t> image(new image_t(width, height, depth, getSpacing()[0] * 1000, getSpacing()[1] * 1000, getSpacing()[2] * 1000));
  std::copy(tensor.begin(), tensor.end(), image->begin());

  newState->setImage(image);
}

} /* namespace convrbm4d */
} /* namespace gml */
