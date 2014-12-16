/*
 * CompareTensor.cpp
 *
 *  Created on: Dec 7, 2014
 *      Author: tombr
 */

#include "CompareTensor.h"

#include <tbblas/math.hpp>
#include <tbblas/sum.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(CompareTensor)

  ReflectableBase(DefaultWorkflowElement<CompareTensor>)

  WorkflowProperty(Input, Input("In"), NotNull<Type>())
  WorkflowProperty(Gold, Input("G"), NotNull<Type>())
  WorkflowProperty(Measure, Enumerator<Type>())
  WorkflowProperty(Value, Output())

EndPropertyDefinitions

CompareTensor::CompareTensor() : _Value(0) {
  setLabel("Compare");
}

void CompareTensor::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  host_tensor_t& input = *getInput();
  host_tensor_t& gold = *getGold();

  double value;

  switch (getMeasure()) {
    case SimilarityMeasure::Sensitivity:              value = sum((input > 0.5) * (gold > 0.5)) / sum(gold > 0.5);                            break;
    case SimilarityMeasure::Specificity:              value = sum((input < 0.5) * (gold < 0.5)) / sum(gold < 0.5);                            break;
    case SimilarityMeasure::DiceCoefficient:          value = 2 * sum ((input > 0.5) * (gold > 0.5)) / (sum(input > 0.5) + sum(gold > 0.5));  break;
    case SimilarityMeasure::PositivePredictiveValue:  value = sum((input > 0.5) * (gold > 0.5)) / sum(input > 0.5);                           break;
    default:
      dlog(Severity::Warning) << "Unsupported measure: " << getMeasure() << ". Aborting!";
      return;
  }

  newState->setValue(value);
}

} /* namespace imageprocessing */

} /* namespace gml */
