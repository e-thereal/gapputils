/*
 * PlanModel.cpp
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#include "PlanModel.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(PlanModel)

  ReflectableBase(DefaultWorkflowElement<PlanModel>)

  WorkflowProperty(InputSize, NotEmpty<Type>(), Description("Must contain the width, height, and depth."))
  WorkflowProperty(StrideWidth, NotEmpty<Type>())
  WorkflowProperty(StrideHeight, NotEmpty<Type>())
  WorkflowProperty(StrideDepth, NotEmpty<Type>())
  WorkflowProperty(FilterWidth, NotEmpty<Type>(), Description("User a filter size of 1 for circular convolutions."))
  WorkflowProperty(FilterHeight, NotEmpty<Type>(), Description("User a filter size of 1 for circular convolutions."))
  WorkflowProperty(FilterDepth, NotEmpty<Type>(), Description("User a filter size of 1 for circular convolutions."))
  WorkflowProperty(OutputWidth, NoParameter())
  WorkflowProperty(OutputHeight, NoParameter())
  WorkflowProperty(OutputDepth, NoParameter())

EndPropertyDefinitions

PlanModel::PlanModel() {
  setLabel("Plan");
}

void PlanModel::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getInputSize().size() != 3) {
    dlog(Severity::Warning) << "The input size must contain the width, height, and depth of the input image.";
    return;
  }

  double width = getInputSize()[0], height = getInputSize()[1], depth = getInputSize()[2];
  size_t layerCount = _StrideWidth.size();
  layerCount = std::min(layerCount, _StrideHeight.size());
  layerCount = std::min(layerCount, _StrideDepth.size());
  layerCount = std::min(layerCount, _FilterWidth.size());
  layerCount = std::min(layerCount, _FilterHeight.size());
  layerCount = std::min(layerCount, _FilterDepth.size());

  std::vector<double> widths, heights, depths;

  for (size_t i = 0; i < layerCount; ++i) {
    widths.push_back(width = ((width - (double)_FilterWidth[i] + (double)_StrideWidth[i]) / (double)_StrideWidth[i]));
    heights.push_back(height = ((height - (double)_FilterHeight[i] + (double)_StrideHeight[i]) / (double)_StrideHeight[i]));
    depths.push_back(depth = ((depth - (double)_FilterDepth[i] + (double)_StrideDepth[i]) / (double)_StrideDepth[i]));
  }

  newState->setOutputWidth(widths);
  newState->setOutputHeight(heights);
  newState->setOutputDepth(depths);
}

} /* namespace convrbm4d */

} /* namespace gml */
