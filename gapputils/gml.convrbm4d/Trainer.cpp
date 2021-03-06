/*
 * Trainer.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer.h"

#include <gapputils/attributes/GroupAttribute.h>

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Trainer, Description("Trains a convolutional RBM using CD-n learning. Can also be used for layer-wise pretraining of DBMs (see the DbmLayer parameter)."))

  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(DbmLayer, Enumerator<Type>())

  WorkflowProperty(EpochCount, Group("Optimization"), Description("Number of sweeps through the entire training set."))
  WorkflowProperty(TrialEpochCount, Group("Optimization"))
  WorkflowProperty(BatchSize, Group("Optimization"), Description("Number of images used per gradient update."))

  WorkflowProperty(SparsityMethod, Group("Regularization"), Enumerator<Type>())
  WorkflowProperty(SparsityTarget, Group("Regularization"), Description("Target expected activation of a hidden unit."))
  WorkflowProperty(SparsityWeight, Group("Regularization"), Description("Weight of the sparsity target relative to the learning rate."))

  WorkflowProperty(FilterBatchSize, Group("Performance"), Description("Number of filters that are processed in parallel."))
  WorkflowProperty(SubRegionCount, Group("Performance"), Description("Number of sub-regions into which the calculation will be split. Fewer (but larger) sub-regions speed up the calculation but require more memory."))
  WorkflowProperty(GpuCount, Group("Performance"), Description("Number of GPUs used for training."))

  WorkflowProperty(CdIterations, Group("Optimization"), Description("Number of CD iterations. (1: CD learning, >1: CD-n learning)"))
  WorkflowProperty(Method, Group("Optimization"), Enumerator<Type>())
  WorkflowProperty(LearningRates, Group("Optimization"), NotEmpty<Type>(), Description("Initial values of the decaying learning rates of the filters."))
  WorkflowProperty(LearningDecay, Group("Optimization"), Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(InitialMomentum, Group("Optimization"), Description("Momentum used for the first epoch."))
  WorkflowProperty(FinalMomentum, Group("Optimization"))
  WorkflowProperty(MomentumDecayEpochs, Group("Optimization"))
  WorkflowProperty(WeightDecay, Group("Regularization"))
  WorkflowProperty(InitialWeights, Group("Optimization"), Description("If given, these weights will be tested as initial weights and will override the initial weights."))
  WorkflowProperty(SignalToNoiseRatio, Group("Optimization"), Description("If greater than 0, the standard deviation of the noise of the initial filters is set to 'sd(learned filters) / SNR'."))
  WorkflowProperty(RandomizeTraining, Group("Optimization"), Flag(), Description("Randomly select images of a mini-batch."))

  WorkflowProperty(ShareBiasTerms, Flag(), Description("If checked, visible and hidden units of the same filter share bias terms."))
  WorkflowProperty(ChannelsPerBlock, Description("All channels of the same pooling block share the same bias terms when shared bias terms are active. Hence, this number must be known."))
  WorkflowProperty(DropoutMethod, Group("Regularization"), Enumerator<Type>(), Description("Defines if entire columns or individual hidden units are dropped."))
  WorkflowProperty(VisibleDropout, Group("Regularization"), Description("Probability of a visible unit of being ignored. (currently not used)"))
  WorkflowProperty(HiddenDropout, Group("Regularization"), Description("Probability of a hidden unit of being ignored."))
  WorkflowProperty(FilterDropout, Group("Regularization"), Description("Probability of an entire batch of filters being ignored. To drop individual filters, set the filter batch size to 1"))
  WorkflowProperty(CalculateError, Flag(), Description("If checked, the reconstruction error is calculated"))
  WorkflowProperty(UpdateModel, Description("If greater than 0, the model is updated every <UpdateModel> epochs."))

  WorkflowProperty(InitialModel, Group("Input/output"), Input("CRBM"), NotNull<Type>(), Description("Required initial model. The initial model can be a previously trained model, are a new model initialized using Initialize"))
  WorkflowProperty(Tensors, Group("Input/output"), Input("Ts"), NotNull<Type>(), NotEmpty<Type>(), Description("The training set."))
  WorkflowProperty(CurrentEpoch, Group("Input/output"), NoParameter())
  WorkflowProperty(Model, Group("Input/output"), Output("CRBM"), Description("The trained model."))
  WorkflowProperty(AverageEpochTime, Group("Input/output"), Output("T"))
  WorkflowProperty(ReconstructionError, Group("Input/output"), NoParameter())

EndPropertyDefinitions

Trainer::Trainer()
 : _EpochCount(100), _TrialEpochCount(10), _BatchSize(20), _FilterBatchSize(1), _SubRegionCount(tbblas::seq<4>(1)), _GpuCount(1),
   _SparsityTarget(1e-2), _SparsityWeight(0.1),
   _CdIterations(1), _LearningDecay(-1), _InitialMomentum(0.5), _FinalMomentum(0.9),
   _MomentumDecayEpochs(20), _WeightDecay(0), _SignalToNoiseRatio(20), _RandomizeTraining(false),
   _ShareBiasTerms(false), _ChannelsPerBlock(1), _VisibleDropout(0.0), _HiddenDropout(0.5), _FilterDropout(0.0),
   _CalculateError(false), _UpdateModel(0),
   _CurrentEpoch(0), _AverageEpochTime(0.0), _ReconstructionError(0.0)
{
  setLabel("Trainer");
  _LearningRates.push_back(1e-3);
}

TrainerChecker trainerChecker;

} /* namespace convrbm */

} /* namespace gml */
