/*
 * Specificity.cpp
 *
 *  Created on: Feb 28, 2013
 *      Author: tombr
 */

#include "Specificity.h"

#include <tbblas/math.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/io.hpp>
#include <cmath>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Specificity)

  ReflectableBase(DefaultWorkflowElement<Specificity>)

  WorkflowProperty(GeneratedTensors, Input("Gen"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Dataset, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(ErrorMeasure, Enumerator<Type>())
  WorkflowProperty(Minimum)
  WorkflowProperty(Maximum)
  WorkflowProperty(AverageError, NoParameter())

EndPropertyDefinitions

Specificity::Specificity() : _Minimum(0), _Maximum(1), _AverageError(0) {
  setLabel("SPE");
}

void Specificity::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  std::vector<boost::shared_ptr<tensor_t> >& tensors = *getGeneratedTensors();
  std::vector<boost::shared_ptr<tensor_t> >& dataset = *getDataset();

  double cumError = 0;
  for (size_t iTensor = 0; iTensor < tensors.size(); ++iTensor) {
    tensor_t tensor = min(getMaximum(), max(getMinimum(), *tensors[iTensor]));
    double minError = 0;
    for (size_t i = 0; i < dataset.size(); ++i) {
      tensor_t& sample = *dataset[i];
      switch (getErrorMeasure()) {
      case SpecificityErrorMeasure::RMSE:
        {
          double error = std::sqrt(dot(tensor - sample, tensor - sample) / tensor.count());
          if (i == 0 || error < minError)
            minError = error;
        }
        break;

      case SpecificityErrorMeasure::CC:
        {
          double meanTensor = sum(tensor) / tensor.count();
          double meanSample = sum(sample) / sample.count();
          double cov = dot(tensor - meanTensor, sample - meanSample);
          double var1 = sqrt(dot(tensor - meanTensor, tensor - meanTensor));
          double var2 = sqrt(dot(sample - meanSample, sample - meanSample));
          double error = cov / (var1 * var2);

//          tbblas_print(meanTensor);
//          tbblas_print(meanSample);
//          tbblas_print(cov);
//          tbblas_print(var1);
//          tbblas_print(var2);
//          tbblas_print(error);

          if (i == 0 || error > minError)
            minError = error;
        }
        break;
      }
    }
    cumError += minError;
    if (monitor)
      monitor->reportProgress(100.0 * iTensor / tensors.size());
  }

  newState->setAverageError(cumError / tensors.size());
}

} /* namespace convrbm4d */
} /* namespace gml */
