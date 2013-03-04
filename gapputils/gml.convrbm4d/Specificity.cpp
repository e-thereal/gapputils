/*
 * Specificity.cpp
 *
 *  Created on: Feb 28, 2013
 *      Author: tombr
 */

#include "Specificity.h"

#include <tbblas/math.hpp>
#include <tbblas/dot.hpp>
#include <cmath>

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(Specificity)

  ReflectableBase(DefaultWorkflowElement<Specificity>)

  WorkflowProperty(GeneratedTensors, Input("Gen"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Dataset, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Minimum)
  WorkflowProperty(Maximum)
  WorkflowProperty(AverageRmse, NoParameter())

EndPropertyDefinitions

Specificity::Specificity() : _Minimum(0), _Maximum(1), _AverageRmse(0) {
  setLabel("SPE");
}

void Specificity::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getGeneratedTensors();
  std::vector<boost::shared_ptr<tensor_t> >& dataset = *getDataset();

  double cumRmse = 0;
  for (size_t iTensor = 0; iTensor < tensors.size(); ++iTensor) {
    tensor_t tensor = min(getMaximum(), max(getMinimum(), *tensors[iTensor]));
    double minRmse = 0;
    for (size_t i = 0; i < dataset.size(); ++i) {
      tensor_t& sample = *dataset[i];
      double rmse = std::sqrt(dot(tensor - sample, tensor - sample) / tensor.count());
      if (i == 0 || rmse < minRmse)
        minRmse = rmse;
    }
    cumRmse += minRmse;
    if (monitor)
      monitor->reportProgress(100.0 * iTensor / tensors.size());
  }

  newState->setAverageRmse(cumRmse / tensors.size());
}

} /* namespace convrbm4d */
} /* namespace gml */
