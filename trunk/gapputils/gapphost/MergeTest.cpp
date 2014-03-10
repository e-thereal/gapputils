/*
 * MergeTest.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#include "MergeTest.h"

#include <capputils/attributes/MergeAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

namespace gapputils {

namespace testing {

#ifdef _RELEASE
BeginPropertyDefinitions(MergeTest, Deprecated("Only available in debug mode."))
#else
BeginPropertyDefinitions(MergeTest)
#endif

  ReflectableBase(DefaultWorkflowElement<MergeTest>)

  WorkflowProperty(Inputs, Input("Ds"), NotNull<Type>(), Merge<Type>())
  WorkflowProperty(Inputs2, Input("Ds"))
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
