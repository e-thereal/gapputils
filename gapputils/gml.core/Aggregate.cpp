/*
 * Aggregate.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: tombr
 */

#include "Aggregate.h"

#include <cmath>
#include <algorithm>

namespace gml {
namespace core {

BeginPropertyDefinitions(Aggregate)

  ReflectableBase(DefaultWorkflowElement<Aggregate>)

  WorkflowProperty(Data, Input(""), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Operation, Enumerator<Type>())
  WorkflowProperty(Value, Output(""))

EndPropertyDefinitions

Aggregate::Aggregate() : _Operation(AggregatorOperation::Sum) {
  setLabel("Sum");
}

void Aggregate::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  std::vector<double>& data = *getData();
  double result = 0;

  switch (getOperation()) {
  case AggregatorOperation::Minimum:
    if (data.size())
      result = data[0];
    for (size_t i = 0; i < data.size(); ++i)
      result = std::min(result, data[i]);
    break;

  case AggregatorOperation::Maximum:
    if (data.size())
      result = data[0];
    for (size_t i = 0; i < data.size(); ++i)
      result = std::max(result, data[i]);
    break;

  case AggregatorOperation::Sum:
    for (size_t i = 0; i < data.size(); ++i)
      result += data[i];
    break;

  case AggregatorOperation::Average:
    for (size_t i = 0; i < data.size(); ++i)
      result += data[i];
    result /= data.size();
    break;

  default:
    dlog(Severity::Warning) << "Unsupported aggregation operation '" << getOperation() << "'. Aborting!";
    return;
  }

  newState->setValue(result);
}

} /* namespace core */

} /* namespace gml */
