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

  WorkflowProperty(EpochCount, Description("Number of sweeps through the entire training set."))
  WorkflowProperty(BatchSize, Description("Number of images used per gradient update."))
  WorkflowProperty(GpuCount, Description("Number of GPUs used for training."))

  WorkflowProperty(SparsityTarget, Description("Target expected activation of a hidden unit."))
  WorkflowProperty(SparsityWeight, Description("Weight of the sparsity target relative to the learning rate."))
  WorkflowProperty(SparsityMethod, Enumerator<Type>())

  WorkflowProperty(LearningRate, Description("Initial value of the exponentially decaying learning rate."))
  WorkflowProperty(LearningDecay, Description("Rate at which the learning rate decays."))
  WorkflowProperty(InitialMomentum, Description("Momentum used for the first epoch."))
  WorkflowProperty(FinalMomentum)
  WorkflowProperty(MomentumDecayEpochs)
  WorkflowProperty(WeightDecay)
  WorkflowProperty(WeightVectorLimit, Description("Maximum length of the weight update vector. Lower values reduce oscillation."))
  WorkflowProperty(RandomizeTraining, Description("Randomly select images of a mini-batch."))
  WorkflowProperty(ShareBiasTerms, Description("If 1, visible and hidden units of the same filter share bias terms."))
  WorkflowProperty(VisibleDropout, Description("Probability of a visible unit of being ignored."))
  WorkflowProperty(HiddenDropout, Description("Probability of a hidden unit of being ignored."))
  WorkflowProperty(FilterDropout, Description("Probability of an entire filter of being ignored."))
  WorkflowProperty(DropoutMethod, Enumerator<Type>(), Description("Defines if entire columns or individual hidden units are dropped."))
  WorkflowProperty(DropoutStage, Enumerator<Type>(), Description("Defines at which stage the dropout decision is made."))
  WorkflowProperty(CalculateError, Description("If 1, the reconstruction error is calculated"))
  WorkflowProperty(UpdateModel, Description("If greater than 0, the model is updated every <UpdateModel> epochs."))

  WorkflowProperty(CurrentEpoch, NoParameter())
  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(ModelIncrement, Output("Inc"))
  WorkflowProperty(AverageEpochTime, Output("T"))
  WorkflowProperty(ReconstructionError, NoParameter())

EndPropertyDefinitions

Trainer::Trainer()
 : _EpochCount(100), _BatchSize(20), _GpuCount(1),
   _SparsityTarget(1e-2), _SparsityWeight(0.1),
   _LearningRate(1e-3), _LearningDecay(0.98), _InitialMomentum(0.5), _FinalMomentum(0.9),
   _MomentumDecayEpochs(50), _WeightDecay(0), _WeightVectorLimit(1), _RandomizeTraining(false),
   _ShareBiasTerms(false), _VisibleDropout(0.2), _HiddenDropout(0.5), _FilterDropout(0.0),
   _CalculateError(false), _UpdateModel(0),
   _CurrentEpoch(0), _AverageEpochTime(0.0), _ReconstructionError(0.0)
{
  setLabel("Trainer");
}

TrainerChecker trainerChecker;

} /* namespace convrbm */
} /* namespace gml */
