/*
 * Trainer.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer.h"

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(Trainer)

  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(InitialModel, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(GpuCount, Description("Specifies the number of GPUs used for training."))
  WorkflowProperty(LearningRateW)
  WorkflowProperty(LearningRateVB)
  WorkflowProperty(LearningRateHB)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight, Description("The sparsity weight is relative to the learning rate."))
  WorkflowProperty(SparsityMethod, Enumerator<Type>())
  WorkflowProperty(RandomizeTraining)
  WorkflowProperty(CalculateError)
  WorkflowProperty(ShareBiasTerms)
  WorkflowProperty(Dropout, Description("If 1, dropout is used for training."))
//  WorkflowProperty(Logfile)

  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(AverageEpochTime, Output("T"))

EndPropertyDefinitions

Trainer::Trainer()
 : _EpochCount(100), _BatchSize(20), _GpuCount(1), _LearningRateW(1e-3), _LearningRateVB(1e-3), _LearningRateHB(1e-3),
   _SparsityTarget(1e-2), _SparsityWeight(0.1), _RandomizeTraining(false), _CalculateError(false), _ShareBiasTerms(false),
   _Dropout(0), _AverageEpochTime(0.0)
{
  setLabel("Trainer");
}

TrainerChecker trainerChecker;

} /* namespace convrbm */
} /* namespace gml */
