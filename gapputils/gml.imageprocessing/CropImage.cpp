/*
 * CropImage.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#include "CropImage.h"

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(CropImage)

  ReflectableBase(DefaultWorkflowElement<CropImage>)

  WorkflowProperty(Input, Input(""), NotNull<Type>())
  WorkflowProperty(CroppedSize, NotEmpty<Type>())
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

CropImage::CropImage() {
  setLabel("Crop");
}

void CropImage::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef float value_t;
  typedef tensor<value_t, 3> tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getCroppedSize().size() != 3 || getCroppedSize()[0] <= 0 || getCroppedSize()[1] <= 0 || getCroppedSize()[2] <= 0) {
    dlog(Severity::Warning) << "Cropped size must contain the width, height, and depth, and all dimensions must be positive. Aborting!";
    return;
  }

  dim_t croppedSize = seq(getCroppedSize()[0], getCroppedSize()[1], getCroppedSize()[2]);
  image_t& input = *getInput();
  tensor_t padded(input.getSize()[0], input.getSize()[1], input.getSize()[2]);
  std::copy(input.begin(), input.end(), padded.begin());
  dim_t topleft = padded.size() / 2 - croppedSize / 2;

  tensor_t cropped = padded[topleft, croppedSize];
  boost::shared_ptr<image_t> output(new image_t(croppedSize[0], croppedSize[1], croppedSize[2],
      input.getPixelSize()[0], input.getPixelSize()[1], input.getPixelSize()[2]));
  std::copy(cropped.begin(), cropped.end(), output->begin());

  newState->setOutput(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
