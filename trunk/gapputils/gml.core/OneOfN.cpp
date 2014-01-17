/*
 * OneOfN.cpp
 *
 *  Created on: Dec 13, 2013
 *      Author: tombr
 */

#include "OneOfN.h"

#include <cmath>

namespace gml {

namespace core {

BeginPropertyDefinitions(OneOfN)

  ReflectableBase(DefaultWorkflowElement<OneOfN>)

  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(LevelCount)
  WorkflowProperty(Outputs, Output("Out"))
  WorkflowProperty(Minimum, NoParameter())
  WorkflowProperty(Maximum, NoParameter())

EndPropertyDefinitions

OneOfN::OneOfN() : _LevelCount(2), _Minimum(0), _Maximum(1) {
  setLabel("1-of-n");
}

void OneOfN::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  // Get the minimum and the maximum and divide the data in equally spaced bins

  v_data_t& inputs = *getInputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->size() != 1) {
      dlog(Severity::Warning) << "All input vectors must be of size 1. Aborting!";
      return;
    }
  }

  double minimum = inputs[0]->at(0), maximum = inputs[0]->at(0);
  for (size_t i = 0; i < inputs.size(); ++i) {
    minimum = std::min(minimum, inputs[i]->at(0));
    maximum = std::max(maximum, inputs[i]->at(0));
  }

  boost::shared_ptr<v_data_t> outputs(new v_data_t());
  for (size_t i = 0; i < inputs.size(); ++i) {
    boost::shared_ptr<data_t> output(new data_t(getLevelCount()));
    double value = inputs[i]->at(0);
    for (size_t j = 0; j < output->size(); ++j) {
      const double lower = (double)j / (double)getLevelCount() * maximum + (double)(getLevelCount() - j) / (double)getLevelCount() * minimum;
      const double upper = (double)(j + 1) / (double)getLevelCount() * maximum + (double)(getLevelCount() - j - 1) / (double)getLevelCount() * minimum;
      if (j == 0)
        output->at(j) = lower <= value && value <= upper;
      else
        output->at(j) = lower < value && value <= upper;
    }
    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
  newState->setMinimum(minimum);
  newState->setMaximum(maximum);
}

} /* namespace core */

} /* namespace gml */
