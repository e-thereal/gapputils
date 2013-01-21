/*
 * AdjustIntensities.cpp
 *
 *  Created on: Oct 26, 2012
 *      Author: tombr
 */

#include "AdjustIntensities.h"

#include <algorithm>
#include <cmath>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(AdjustIntensities)

  ReflectableBase(DefaultWorkflowElement<AdjustIntensities>)

  WorkflowProperty(Input, Input("Img"), NotNull<Type>())
  WorkflowProperty(InputLowerBound)
  WorkflowProperty(InputUpperBound)
  WorkflowProperty(InputIntensities, Input("In"))
  WorkflowProperty(OutputLowerBound)
  WorkflowProperty(OutputUpperBound)
  WorkflowProperty(OutputIntensities, Input("Out"))
  WorkflowProperty(Output, Output("Img"))

EndPropertyDefinitions

AdjustIntensities::AdjustIntensities() : _InputLowerBound(0.0), _InputUpperBound(1.0), _OutputLowerBound(0.0), _OutputUpperBound(1.0) {
  setLabel("Adjust");
}

void AdjustIntensities::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  image_t& input = *getInput();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

  int inputCount = 2, outputCount = 2;
  if (getInputIntensities())
    inputCount += getInputIntensities()->size();
  if (getOutputIntensities())
    outputCount += getOutputIntensities()->size();

  std::vector<float> inIntens(inputCount);
  std::vector<float> outIntens(outputCount);

  if (inputCount != outputCount) {
    dlog(Severity::Warning) << "Input and output intensity sizes don't match. Aborting!";
    return;
  }

  inIntens[0] = (float)getInputLowerBound();
  outIntens[0] = (float)getOutputLowerBound();
  inIntens[inIntens.size()-1] = (float)getInputUpperBound();
  outIntens[outIntens.size()-1] = (float)getOutputUpperBound();
  if (getInputIntensities())
    std::copy(getInputIntensities()->begin(), getInputIntensities()->end(), inIntens.begin() + 1);
  if (getOutputIntensities())
    std::copy(getOutputIntensities()->begin(), getOutputIntensities()->end(), outIntens.begin() + 1);

  float* idata = input.getData();
  float* odata = output->getData();
  size_t count = input.getCount();

  for (size_t i = 0; i < count && (monitor ? !monitor->getAbortRequested() : true); ++i) {
    double inValue = std::max(inIntens[0], std::min(inIntens[inIntens.size()-1], idata[i]));
    
    for (size_t j = 1; j < inIntens.size(); ++j) {
      if (inValue <= inIntens[j] || j == inIntens.size() - 1) {
        odata[i] = (inValue - inIntens[j-1]) / (inIntens[j] - inIntens[j-1]) *
            (outIntens[j] - outIntens[j-1]) + outIntens[j-1];
        break;
      }
    }
    if ((i * 100) % count == 0 && monitor)
      monitor->reportProgress(100.0 * i / count);
  }

  newState->setOutput(output);
}

}

} /* namespace gml */
