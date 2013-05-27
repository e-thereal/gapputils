/*
 * SplitModel.cpp
 *
 *  Created on: Jan 9, 2013
 *      Author: tombr
 */

#include "SplitModel.h"

#include <cmath>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(SplitModel)

  ReflectableBase(DefaultWorkflowElement<SplitModel>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(MaxFilterCount, Description("A value of -1 indicates all filters."))

  WorkflowProperty(Filters, Output("F"))
  WorkflowProperty(VisibleBias, Output("B"))
  WorkflowProperty(HiddenBiases, Output("H"))
  WorkflowProperty(FilterKernelSize, NoParameter())
  WorkflowProperty(Mean, NoParameter())
  WorkflowProperty(Stddev, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())

EndPropertyDefinitions

SplitModel::SplitModel() : _MaxFilterCount(-1) {
  setLabel("SplitModel");
}

void SplitModel::update(IProgressMonitor* monitor) const {
  Model& model = *getModel();

  const int filterCount = std::min(getMaxFilterCount(), (int)model.getFilters()->size());

  if (filterCount > 0) {
    auto filters = boost::make_shared<std::vector<boost::shared_ptr<tensor_t> > >(filterCount);
    auto biases = boost::make_shared<std::vector<boost::shared_ptr<tensor_t> > >(filterCount);
    std::copy(model.getFilters()->begin(), model.getFilters()->begin() + filterCount, filters->begin());
    std::copy(model.getHiddenBiases()->begin(), model.getHiddenBiases()->begin() + filterCount, biases->begin());
    newState->setFilters(filters);
    newState->setHiddenBiases(biases);
  } else {
    newState->setFilters(model.getFilters());
    newState->setHiddenBiases(model.getHiddenBiases());
  }
  newState->setVisibleBias(model.getVisibleBias());
  newState->setFilterKernelSize(model.getFilterKernelSize());
  newState->setMean(model.getMean());
  newState->setStddev(model.getStddev());
  newState->setVisibleUnitType(model.getVisibleUnitType());
  newState->setHiddenUnitType(model.getHiddenUnitType());
}

} /* namespace convrbm4d */

} /* namespace gml */
