/*
 * Jaccobian.cpp
 *
 *  Created on: Nov 26, 2014
 *      Author: tombr
 */

#include "Jaccobian.h"

#include <tbblas/zeros.hpp>

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(Jaccobian)

  ReflectableBase(DefaultWorkflowElement<Jaccobian>)

  WorkflowProperty(Input, Input("T"))
  WorkflowProperty(Inputs, Input("Ts"))
  WorkflowProperty(Output, Output("T"))
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

Jaccobian::Jaccobian() {
  setLabel("Jac");
}

void Jaccobian::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  if (getInput()) {
    host_tensor_t& input = *getInput();

    host_tensor_t::dim_t size = input.size() - seq(2,2,2,0), slice = size, outSize = input.size();
    slice[3] = 1;
    outSize[3] = 1;


    if (size[3] != 3) {
      dlog(Severity::Warning) << "The forth dimension of the vector field must be 3. Aborting!";
      return;
    }

    if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
      dlog(Severity::Warning) << "Input field to small to calculate the Jaccobian. Aborting!";
      return;
    }

    host_tensor_t jacx = input[seq(2,0,0,0), size] - input[seq(0,0,0,0), size];
    host_tensor_t jacy = input[seq(0,2,0,0), size] - input[seq(0,0,0,0), size];
    host_tensor_t jacz = input[seq(0,0,2,0), size] - input[seq(0,0,0,0), size];

    host_tensor_t det =
        jacx[seq(0,0,0,0), slice] * (jacy[seq(0,0,0,1), slice] * jacz[seq(0,0,0,2), slice] - jacy[seq(0,0,0,2), slice] * jacz[seq(0,0,0,1), slice]) -
        jacy[seq(0,0,0,0), slice] * (jacx[seq(0,0,0,1), slice] * jacz[seq(0,0,0,2), slice] - jacx[seq(0,0,0,2), slice] * jacz[seq(0,0,0,1), slice]) +
        jacz[seq(0,0,0,0), slice] * (jacx[seq(0,0,0,1), slice] * jacy[seq(0,0,0,2), slice] - jacx[seq(0,0,0,2), slice] * jacy[seq(0,0,0,1), slice]);


    boost::shared_ptr<host_tensor_t> output(new host_tensor_t(zeros<host_tensor_t::value_t>(outSize)));
    (*output)[seq(1,1,1,0), slice] = det;

    newState->setOutput(output);
  }

  if (getInputs()) {
    v_host_tensor_t& inputs = *getInputs();
    boost::shared_ptr<v_host_tensor_t> outputs(new v_host_tensor_t());

    for (size_t i = 0; i < inputs.size() && (monitor ? !monitor->getAbortRequested() : true); ++i) {
      host_tensor_t& input = *inputs[i];

      host_tensor_t::dim_t size = input.size() - seq(2,2,2,0), slice = size, outSize = input.size();
      slice[3] = 1;
      outSize[3] = 1;

      if (size[3] != 3) {
        dlog(Severity::Warning) << "The forth dimension of the vector field must be 3. Aborting!";
        return;
      }

      if (size[0] < 1 || size[1] < 1 || size[2] < 1) {
        dlog(Severity::Warning) << "Input field to small to calculate the Jaccobian. Aborting!";
        return;
      }

      host_tensor_t jacx = input[seq(2,0,0,0), size] - input[seq(0,0,0,0), size];
      host_tensor_t jacy = input[seq(0,2,0,0), size] - input[seq(0,0,0,0), size];
      host_tensor_t jacz = input[seq(0,0,2,0), size] - input[seq(0,0,0,0), size];

      host_tensor_t det =
          jacx[seq(0,0,0,0), slice] * (jacy[seq(0,0,0,1), slice] * jacz[seq(0,0,0,2), slice] - jacy[seq(0,0,0,2), slice] * jacz[seq(0,0,0,1), slice]) -
          jacy[seq(0,0,0,0), slice] * (jacx[seq(0,0,0,1), slice] * jacz[seq(0,0,0,2), slice] - jacx[seq(0,0,0,2), slice] * jacz[seq(0,0,0,1), slice]) +
          jacz[seq(0,0,0,0), slice] * (jacx[seq(0,0,0,1), slice] * jacy[seq(0,0,0,2), slice] - jacx[seq(0,0,0,2), slice] * jacy[seq(0,0,0,1), slice]);


      boost::shared_ptr<host_tensor_t> output(new host_tensor_t(zeros<host_tensor_t::value_t>(outSize)));
      (*output)[seq(1,1,1,0), slice] = det;

      outputs->push_back(output);

      if (monitor) {
        monitor->reportProgress(100. * (i + 1) / inputs.size());
      }
    }
    newState->setOutputs(outputs);
  }
}

} /* namespace imageprocessing */

} /* namespace gml */
