/*
 * CorrelationTensor.cpp
 *
 *  Created on: Dec 10, 2013
 *      Author: tombr
 */

#include "CorrelationTensor.h"

#include <tbblas/zeros.hpp>
#include <tbblas/math.hpp>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(CorrelationTensor, Description("Calculates the correlation of an input vector with each element of a vector of tensors."))

  ReflectableBase(DefaultWorkflowElement<CorrelationTensor>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Data, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(CorrelationTensor, Output("T"))

EndPropertyDefinitions

CorrelationTensor::CorrelationTensor() {
  setLabel("Corr");
}

void CorrelationTensor::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  v_tensor_t& inputs = *getTensors();
  data_t& data = *getData();

  if (data.size() != inputs.size()) {
    dlog(Severity::Warning) << "Data size and numbers of tensors don't match. Aborting!";
    return;
  }

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->size() != inputs[0]->size()) {
      dlog(Severity::Warning) << "All tensors must be of the same size. Aborting!";
      return;
    }
  }

  boost::shared_ptr<tensor_t> output(new tensor_t(inputs[0]->size()));

  // Calculate means
  tensor_t mean1 = zeros<value_t>(inputs[0]->size());
  value_t mean2 = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    mean1 = mean1 + *inputs[0];
    mean2 = mean2 + data[i];
  }
  mean1 = mean1 / inputs.size();
  mean2 = mean2 / data.size();

  tensor_t cov = zeros<value_t>(mean1.size()), sd1 = zeros<value_t>(mean1.size());
  value_t sd2 = 0;

  for (size_t i = 0; i < inputs.size(); ++i) {
    cov = cov + (*inputs[i] - mean1) * (data[i] - mean2);
    sd1 = sd1 + (*inputs[i] - mean1) * (*inputs[i] - mean1);
    sd2 = sd2 + (data[i] - mean2) * (data[i] - mean2);
  }

  const value_t eps = 1e-7;
  *output = cov / (sqrt(sd1 + eps) * sqrt(sd2 + eps));

  newState->setCorrelationTensor(output);
}

} /* namespace convrbm4d */

} /* namespace gml */
