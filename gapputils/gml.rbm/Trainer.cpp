/*
 * Trainer.cpp
 *
 *  Created on: Jan 14, 2013
 *      Author: tombr
 */

#include "Trainer.h"

namespace gml {

namespace rbm {

BeginPropertyDefinitions(Trainer)

  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(TrainingSet, Input("Data"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(HiddenCount)
  WorkflowProperty(SampleHiddens)
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRate)
  WorkflowProperty(InitialWeights)
  WorkflowProperty(InitialHidden)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight)
  WorkflowProperty(VisibleUnitType, Enumerator<Type>())
  WorkflowProperty(HiddenUnitType, Enumerator<Type>())
  WorkflowProperty(ShowWeights, Description("Only the first ShowWeights features are shown."))
  WorkflowProperty(ShowEvery, Description("Debug output is shown only every ShowEvery epochs."))

  WorkflowProperty(Model, Output("RBM"))
  WorkflowProperty(Weights, Output("W"))

EndPropertyDefinitions

Trainer::Trainer()
 : _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01), _InitialWeights(0.01), _InitialHidden(0.0),
   _SparsityTarget(0.1), _SparsityWeight(0.1), _ShowWeights(0), _ShowEvery(1)
{
  setLabel("Trainer");
}

TrainerChecker trainerChecker;

}

}
