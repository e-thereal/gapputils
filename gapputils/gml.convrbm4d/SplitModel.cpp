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
  model_t& model = *getModel();

  const int filterCount = std::min(getMaxFilterCount(), (int)model.filters().size());

  if (filterCount > 0) {
    boost::shared_ptr<v_tensor_t> filters = boost::make_shared<v_tensor_t>(filterCount);
    boost::shared_ptr<v_tensor_t> biases = boost::make_shared<v_tensor_t>(filterCount);
    std::copy(model.filters().begin(), model.filters().begin() + filterCount, filters->begin());
    std::copy(model.hidden_bias().begin(), model.hidden_bias().begin() + filterCount, biases->begin());
    newState->setFilters(filters);
    newState->setHiddenBiases(biases);
  } else {
    boost::shared_ptr<v_tensor_t> filters = boost::make_shared<v_tensor_t>(model.filters().size());
    boost::shared_ptr<v_tensor_t> biases = boost::make_shared<v_tensor_t>(model.hidden_bias().size());
    std::copy(model.filters().begin(), model.filters().end(), filters->begin());
    std::copy(model.hidden_bias().begin(), model.hidden_bias().end(), biases->begin());
    newState->setFilters(filters);
    newState->setHiddenBiases(biases);
  }
  newState->setVisibleBias(boost::make_shared<tensor_t>(model.visible_bias()));
  newState->setFilterKernelSize(model.kernel_size());
  newState->setMean(model.mean());
  newState->setStddev(model.stddev());
  newState->setVisibleUnitType(model.visibles_type());
  newState->setHiddenUnitType(model.hiddens_type());
}

} /* namespace convrbm4d */

} /* namespace gml */
