/*
 * FineTuning.cpp
 *
 *  Created on: Jul 21, 2014
 *      Author: tombr
 */

#include "FineTuning.h"

namespace gml {

namespace dbn {

BeginPropertyDefinitions(FineTuning, Description("Used for fine-tuning a convDBN."))

  ReflectableBase(DefaultWorkflowElement<FineTuning>)

  WorkflowProperty(InitialModel, Input("DBN"), NotNull<Type>(), Description("Required initial model. The initial model can be a previously trained model, are a new model initialized using CreateDbn"))
  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>(), Description("The training set."))

  WorkflowProperty(EpochCount, Description("Number of sweeps through the entire training set."))
  WorkflowProperty(BatchSize, Description("Number of images used per gradient update."))
  WorkflowProperty(GpuCount, Description("Number of GPUs used for training."))
  WorkflowProperty(FilterBatchLength)

  WorkflowProperty(LearningRate, Description("Initial value of the learning rate."))
  WorkflowProperty(InitialMomentum, Description("Momentum used for the first epoch."))
  WorkflowProperty(FinalMomentum)
  WorkflowProperty(MomentumDecayEpochs)
  WorkflowProperty(WeightDecay)
  WorkflowProperty(RandomizeTraining, Flag(), Description("Randomly select images of a mini-batch."))

  WorkflowProperty(Model, Output("DBN"), Description("The trained model."))
  WorkflowProperty(ReconstructionError, NoParameter())

EndPropertyDefinitions

FineTuning::FineTuning()
 : _EpochCount(100), _BatchSize(20), _GpuCount(1),
   _LearningRate(1e-4), _InitialMomentum(0.5), _FinalMomentum(0.9), _MomentumDecayEpochs(20),
   _WeightDecay(0), _RandomizeTraining(false), _ReconstructionError(0)
{
  setLabel("FineTuning");
}

FineTuningChecker fineTuningChecker;

} /* namespace dbn */

} /* namespace gml */
