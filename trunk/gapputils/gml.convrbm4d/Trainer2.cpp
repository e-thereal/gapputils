/*
 * Trainer2.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer2.h"

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(Trainer2)

  ReflectableBase(DefaultWorkflowElement<Trainer2>)

  WorkflowProperty(InitialModel, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(GpuCount, Description("Specifies the number of GPUs used for training."))
  WorkflowProperty(FilterMethod, Enumerator<Type>())
  WorkflowProperty(Stride, Description("The stride is only used when ConvNet is selected as the filter method."))
  WorkflowProperty(LearningRateW)
  WorkflowProperty(LearningRateVB)
  WorkflowProperty(LearningRateHB)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight, Description("The sparsity weight is relative to the learning rate."))
  WorkflowProperty(SparsityMethod, Enumerator<Type>())
  WorkflowProperty(RandomizeTraining)
  WorkflowProperty(CalculateError)
  WorkflowProperty(ShareBiasTerms)
//  WorkflowProperty(Logfile)

  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(AverageEpochTime, Output("T"))

EndPropertyDefinitions

Trainer2::Trainer2()
 : _EpochCount(100), _BatchSize(20), _GpuCount(1), _Stride(1), _LearningRateW(1e-3), _LearningRateVB(1e-3), _LearningRateHB(1e-3),
   _SparsityTarget(1e-2), _SparsityWeight(0.1), _RandomizeTraining(false), _CalculateError(false), _ShareBiasTerms(false),
   _AverageEpochTime(0.0)
{
  setLabel("Trainer2");
}

Trainer2Checker trainer2Checker;

} /* namespace convrbm */
} /* namespace gml */
