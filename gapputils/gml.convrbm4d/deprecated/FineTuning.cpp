/*
 * FineTuning.cpp
 *
 *  Created on: Jul 16, 2013
 *      Author: tombr
 */

#include "FineTuning.h"

#include <capputils/DeprecatedAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(FineTuning, Deprecated("Use gml.dbm.FineTuning instead."))

  ReflectableBase(DefaultWorkflowElement<FineTuning>)

  WorkflowProperty(Dataset, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(InitialModel, Input("DBM"), NotNull<Type>())
  WorkflowProperty(GpuCount)
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRate)
  WorkflowProperty(LearningDecay, Description("Number of epochs until the learning rate is half the initial learning rate."))
  WorkflowProperty(MeanFieldIterations)
  WorkflowProperty(GibbsIterations)
  WorkflowProperty(SampleCount)
  WorkflowProperty(OutputModel, Output("DBM"))

EndPropertyDefinitions

FineTuning::FineTuning()
 : _GpuCount(1), _EpochCount(1), _BatchSize(20), _LearningRate(1e-5), _LearningDecay(20),
   _MeanFieldIterations(5), _GibbsIterations(5), _SampleCount(20)
{
  setLabel("FineTuning");
}

FineTuningChecker fineTuningChecker;

} /* namespace convrbm4d */

} /* namespace gml */