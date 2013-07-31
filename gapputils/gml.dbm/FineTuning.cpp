/*
 * FineTuning.cpp
 *
 *  Created on: Jul 16, 2013
 *      Author: tombr
 */

#include "FineTuning.h"

namespace gml {

namespace dbm {

BeginPropertyDefinitions(FineTuning)

  ReflectableBase(DefaultWorkflowElement<FineTuning>)

  WorkflowProperty(Dataset, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(InitialModel, Input("DBM"), NotNull<Type>())
  WorkflowProperty(GpuCount)
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRateCL)
  WorkflowProperty(LearningRateRL)
  WorkflowProperty(LearningDecay, Description("Number of epochs until the learning rate is half the initial learning rate."))
  WorkflowProperty(MeanFieldIterations)
  WorkflowProperty(InitialGibbsIterations)
  WorkflowProperty(GibbsIterations)
  WorkflowProperty(SampleCount)
  WorkflowProperty(OutputModel, Output("DBM"))

EndPropertyDefinitions

FineTuning::FineTuning()
 : _GpuCount(1), _EpochCount(1), _BatchSize(20), _LearningRateCL(1e-6), _LearningRateRL(1e-8), _LearningDecay(20),
   _MeanFieldIterations(5), _InitialGibbsIterations(100), _GibbsIterations(5), _SampleCount(20)
{
  setLabel("FineTuning");
}

FineTuningChecker fineTuningChecker;

} /* namespace dbm */

} /* namespace gml */
