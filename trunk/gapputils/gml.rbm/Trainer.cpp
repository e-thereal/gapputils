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
  WorkflowProperty(DbmLayer, Enumerator<Type>())
  WorkflowProperty(Mask, Input("Mask"))
  WorkflowProperty(AutoCreateMask, Flag(), Description("If checked and no mask is given, then visible units that are always 0 are masked out."))
  WorkflowProperty(HiddenCount)
  WorkflowProperty(SampleHiddens, Flag())
  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(LearningRate)
  WorkflowProperty(InitialWeights)
  WorkflowProperty(InitialVisible)
  WorkflowProperty(InitialHidden)
  WorkflowProperty(SparsityTarget)
  WorkflowProperty(SparsityWeight)
  WorkflowProperty(VisibleUnitType, Enumerator<Type>())
  WorkflowProperty(HiddenUnitType, Enumerator<Type>())
  WorkflowProperty(ShowWeights, Description("Only the first ShowWeights features are shown."))
  WorkflowProperty(ShowEvery, Description("Debug output is shown only every ShowEvery epochs."))

  WorkflowProperty(Model, Output("RBM"))
  WorkflowProperty(Weights, Output("W"))
  WorkflowProperty(DebugMask, Output("M"))
  WorkflowProperty(DebugMask2, Output("M2"))

EndPropertyDefinitions

Trainer::Trainer()
 : _AutoCreateMask(false), _HiddenCount(1), _SampleHiddens(true),
   _EpochCount(1), _BatchSize(10), _LearningRate(0.01), _InitialWeights(0.01), _InitialVisible(0.0), _InitialHidden(0.0),
   _SparsityTarget(0.1), _SparsityWeight(0.1), _ShowWeights(0), _ShowEvery(1)
{
  setLabel("Trainer");
}

TrainerChecker trainerChecker;

}

}
