/*
 * SplitModel.cpp
 *
 *  Created on: Jan 9, 2013
 *      Author: tombr
 */

#include "SplitModel.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(SplitModel)

  ReflectableBase(DefaultWorkflowElement<SplitModel>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>())

  WorkflowProperty(Filters, Output("F"))
  WorkflowProperty(VisibleBias, Output("B"))
  WorkflowProperty(HiddenBiases, Output("H"))
  WorkflowProperty(FilterKernelSize, NoParameter())
  WorkflowProperty(Mean, NoParameter())
  WorkflowProperty(Stddev, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())

EndPropertyDefinitions

SplitModel::SplitModel() {
  setLabel("SplitModel");
}

void SplitModel::update(IProgressMonitor* monitor) const {
  Model& model = *getModel();

  newState->setFilters(model.getFilters());
  newState->setVisibleBias(model.getVisibleBias());
  newState->setHiddenBiases(model.getHiddenBiases());
  newState->setFilterKernelSize(model.getFilterKernelSize());
  newState->setMean(model.getMean());
  newState->setStddev(model.getStddev());
  newState->setVisibleUnitType(model.getVisibleUnitType());
  newState->setHiddenUnitType(model.getHiddenUnitType());
}

} /* namespace convrbm4d */

} /* namespace gml */
