/*
 * MergeTest.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#include "MergeTest.h"

#include <capputils/MergeAttribute.h>

namespace gapputils {

namespace testing {

BeginPropertyDefinitions(MergeTest)

  ReflectableBase(DefaultWorkflowElement<MergeTest>)

  WorkflowProperty(Inputs, Input("Ds"), NotNull<Type>(), Merge<Type>())
  WorkflowProperty(Inputs2, Input("Ds"), NotNull<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

MergeTest::MergeTest() {
  setLabel("MergeTest");
}

void MergeTest::update(IProgressMonitor* /*monitor*/) const {
  std::vector<double> outputs(getInputs()->begin(), getInputs()->end());
  newState->setOutputs(outputs);
}

} /* namespace testing */

} /* namespace gapputils */
