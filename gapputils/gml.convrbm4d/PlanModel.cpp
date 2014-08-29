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

}

} /* namespace convrbm4d */

} /* namespace gml */
