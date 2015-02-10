/*
 * AggregateTensors.cpp
 *
 *  Created on: Feb 9, 2015
 *      Author: tombr
 */

#include "AggregateTensors.h"

#include <capputils/attributes/MergeAttribute.h>

#include <tbblas/math.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(AggregateTensors)

  ReflectableBase(DefaultWorkflowElement<AggregateTensors>)

  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(Function, Enumerator<Type>())
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

AggregateTensors::AggregateTensors() {
  setLabel("Aggregate");
}

void AggregateTensors::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  vv_host_tensor_t& inputs = *getInputs();
  boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

  const size_t tensorCount = inputs[0]->size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->size() != tensorCount) {
      dlog(Severity::Warning) << "All input tensor data sets must have the same number of tensors. Aborting!";
      return;
    }
  }

  for (size_t iTensor = 0; iTensor < tensorCount; ++iTensor) {
    boost::shared_ptr<host_tensor_t> output(new host_tensor_t(*inputs[0]->at(iTensor)));
    switch (getFunction()) {
    case AggregatorFunction::Average:
      for (size_t iDataset = 1; iDataset < inputs.size(); ++iDataset) {
        *output = *output + *inputs[iDataset]->at(iTensor);
      }
      *output = *output / inputs.size();
      break;

    case AggregatorFunction::Maximum:
      for (size_t iDataset = 1; iDataset < inputs.size(); ++iDataset) {
        *output = max(*output, *inputs[iDataset]->at(iTensor));
      }
      break;

    case AggregatorFunction::Product:
      for (size_t iDataset = 1; iDataset < inputs.size(); ++iDataset) {
        *output = *output * *inputs[iDataset]->at(iTensor);
      }
      break;

    default:
      dlog(Severity::Warning) << "Unsupported aggregation function. Aborting!";
      return;
    }
    outputs->push_back(output);
  }

  newState->setOutputs(outputs);
}

} /* namespace imageprocessing */

} /* namespace gml */
