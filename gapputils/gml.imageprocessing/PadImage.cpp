/*
 * PadImage.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#include "PadImage.h"

#include <tbblas/tensor.hpp>
#include <tbblas/zeros.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(PadImage)

  ReflectableBase(DefaultWorkflowElement<PadImage>)

  WorkflowProperty(Input, Input(""), NotNull<Type>())
  WorkflowProperty(PaddedSize, NotEmpty<Type>())
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

PadImage::PadImage() {
  setLabel("Pad");
}

void PadImage::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef float value_t;
  typedef tensor<value_t, 3> tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getPaddedSize().size() != 3 || getPaddedSize()[0] <= 0 || getPaddedSize()[1] <= 0 || getPaddedSize()[2] <= 0) {
    dlog(Severity::Warning) << "Padded size must contain the width, height, and depth, and all dimensions must be positive. Aborting!";
    return;
  }

  dim_t padSize = seq(getPaddedSize()[0], getPaddedSize()[1], getPaddedSize()[2]);
  image_t& input = *getInput();
  tensor_t kern(input.getSize()[0], input.getSize()[1], input.getSize()[2]);
  std::copy(input.begin(), input.end(), kern.begin());
  dim_t topleft = padSize / 2 - kern.size() / 2;

  tensor_t padded = zeros<value_t>(padSize);
  padded[topleft, kern.size()] = kern;

  boost::shared_ptr<image_t> output(new image_t(padSize[0], padSize[1], padSize[2],
      input.getPixelSize()[0], input.getPixelSize()[1], input.getPixelSize()[2]));
  std::copy(padded.begin(), padded.end(), output->begin());

  newState->setOutput(output);
}

} /* namespace imageprocessing */

} /* namespace gml */
