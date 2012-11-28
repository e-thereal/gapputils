/*
 * Trainer.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer.h"

namespace gml {
namespace convrbm {

BeginPropertyDefinitions(Trainer)

  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(InitialModel, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRate)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight)

  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(Filters, Output("F"))

EndPropertyDefinitions

Trainer::Trainer()
 : _EpochCount(100), _BatchSize(20), _LearningRate(1e-3),
   _SparsityTarget(1e-2), _SparsityWeight(0)
{
  setLabel("Trainer");
}

Trainer::~Trainer() { }

TrainerChecker trainerChecker;

} /* namespace convrbm */
} /* namespace gml */
