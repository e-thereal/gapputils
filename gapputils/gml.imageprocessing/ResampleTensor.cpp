/*
 * ResampleTensor.cpp
 *
 *  Created on: Feb 5, 2015
 *      Author: tombr
 */

#include "ResampleTensor.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(ResampleTensor)

  ReflectableBase(DefaultWorkflowElement<ResampleTensor>)

  WorkflowProperty(Input, Input("T"))
  WorkflowProperty(Inputs, Input("Ts"))
  WorkflowProperty(Size)
  WorkflowProperty(Output, Output("T"))
  WorkflowProperty(Outputs, Output("Ts"))

EndPropertyDefinitions

ResampleTensor::ResampleTensor() : _Size(tbblas::seq<4>(1)) {
  setLabel("Resample");
}

ResampleTensorChecker resampleTensorChecker;

} /* namespace imageprocessing */

} /* namespace gml */
