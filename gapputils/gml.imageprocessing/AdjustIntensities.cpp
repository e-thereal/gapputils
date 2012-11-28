/*
 * AdjustIntensities.cpp
 *
 *  Created on: Oct 26, 2012
 *      Author: tombr
 */

#include "AdjustIntensities.h"

#include <algorithm>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(AdjustIntensities)

  ReflectableBase(DefaultWorkflowElement<AdjustIntensities>)

  WorkflowProperty(Input, Input("Img"), NotNull<Type>())
  WorkflowProperty(InputIntensities, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(OutputIntensities, Input("Out"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Output, Output("Img"))

EndPropertyDefinitions

AdjustIntensities::AdjustIntensities() {
  setLabel("Adjust");
}

void AdjustIntensities::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  image_t& input = *getInput();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));
  std::vector<double> inIntens(getInputIntensities()->size() + 2);
  std::vector<double> outIntens(getOutputIntensities()->size() + 2);

  if (inIntens.size() != outIntens.size()) {
    dlog(Severity::Warning) << "Input and output intensity sizes don't match. Aborting!";
    return;
  }

  inIntens[0] = outIntens[0] = 0.0;
  inIntens[inIntens.size()-1] = outIntens[outIntens.size()-1] = 1.0;
  std::copy(getInputIntensities()->begin(), getInputIntensities()->end(), inIntens.begin() + 1);
  std::copy(getOutputIntensities()->begin(), getOutputIntensities()->end(), outIntens.begin() + 1);

  float* idata = input.getData();
  float* odata = output->getData();
  size_t count = input.getCount();

  for (size_t i = 0; i < count; ++i) {
    double inValue = idata[i];
    if (inValue < 0.0 || inValue > 1.0) {
      dlog(Severity::Warning) << "Pixel intensity outside [0,1].";
    }
    for (size_t j = 1; j < inIntens.size(); ++j) {
      if (inValue <= inIntens[j] || j == inIntens.size() - 1) {
        odata[i] = (inValue - inIntens[j-1]) / (inIntens[j] - inIntens[j-1]) *
            (outIntens[j] - outIntens[j-1]) + outIntens[j-1];
        break;
      }
    }
  }

  newState->setOutput(output);
}

}

} /* namespace gml */
