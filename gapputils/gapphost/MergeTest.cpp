/*
 * MergeTest.cpp
 *
 *  Created on: Jan 24, 2013
 *      Author: tombr
 */

#include "MergeTest.h"

#include <capputils/attributes/MergeAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

namespace debug {

#ifdef _RELEASE
BeginPropertyDefinitions(MergeTest, Deprecated("Only available in debug mode."))
#else
BeginPropertyDefinitions(MergeTest)
#endif

  ReflectableBase(DefaultWorkflowElement<MergeTest>)

  WorkflowProperty(Inputs, Input("Ds"), Merge<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

MergeTest::MergeTest() {
  setLabel("MergeTest");
}

void MergeTest::update(IProgressMonitor* /*monitor*/) const {
  std::vector<double> outputs(_Inputs.begin(), _Inputs.end());
  newState->setOutputs(outputs);
}

} /* namespace debug */
