/*
 * ReconstructionTest.cpp
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#include "ReconstructionTest.h"

namespace gml {

namespace dbn {

BeginPropertyDefinitions(ReconstructionTest)

  ReflectableBase(DefaultWorkflowElement<ReconstructionTest>)

  WorkflowProperty(Model, Input("Dbn"), NotNull<Type>())
  WorkflowProperty(Dataset, Input("Data"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Type, Enumerator<Type>())
  WorkflowProperty(MaxLayer, Description("Layer from which the reconstructions will be calculated. A value of 0 indicates the visible units. A value of -1 indicates the top-most layer."))
  WorkflowProperty(GpuCount)
  WorkflowProperty(FilterBatchLength)
  WorkflowProperty(Reconstructions, Output("Recon"))
  WorkflowProperty(ReconstructionError, NoParameter())

EndPropertyDefinitions

ReconstructionTest::ReconstructionTest() : _MaxLayer(-1), _GpuCount(1), _ReconstructionError(0) {
  setLabel("Reconstruct");
}

ReconstructionTestChecker reconstructionTestChecker;

} /* namespace dbn */

} /* namespace gml */
