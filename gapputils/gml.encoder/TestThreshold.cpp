/*
 * TestThreshold.cpp
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#include "TestThreshold.h"

namespace gml {

namespace encoder {

BeginPropertyDefinitions(TestThreshold)

  ReflectableBase(DefaultWorkflowElement<TestThreshold>)

  WorkflowProperty(InitialModel, Input("ENN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("TS"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TrainingLabels, Input("TL"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TestSet, Input("ES"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(TestLabels, Input("EL"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Metric, Enumerator<Type>())
  WorkflowProperty(FilterBatchSize)
  WorkflowProperty(SubRegionCount, Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(GlobalTPR, Output("GTPR"))
  WorkflowProperty(GlobalPPV, Output("GPPV"))
  WorkflowProperty(GlobalDSC, Output("GDSC"))
  WorkflowProperty(OptimalTPR, Output("OTPR"))
  WorkflowProperty(OptimalPPV, Output("OPPV"))
  WorkflowProperty(OptimalDSC, Output("ODSC"))
  WorkflowProperty(PredictedTPR, Output("PTPR"))
  WorkflowProperty(PredictedPPV, Output("PPPV"))
  WorkflowProperty(PredictedDSC, Output("PDSC"))

EndPropertyDefinitions

TestThreshold::TestThreshold() : _SubRegionCount(tbblas::seq<host_tensor_t::dimCount>(1)) {
  setLabel("Test");
}

TestThresholdChecker testThresholdChecker;

} /* namespace nn */

} /* namespace gml */
