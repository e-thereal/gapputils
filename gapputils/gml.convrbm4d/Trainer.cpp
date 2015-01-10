/*
 * Trainer.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "Trainer.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(Trainer, Description("Trains a convolutional RBM using CD-n learning. Can also be used for layer-wise pretraining of DBMs (see the DbmLayer parameter)."))

  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(InitialModel, Input("CRBM"), NotNull<Type>(), Description("Required initial model. The initial model can be a previously trained model, are a new model initialized using Initialize"))
  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>(), Description("The training set."))
  WorkflowProperty(DbmLayer, Enumerator<Type>())

  WorkflowProperty(EpochCount, Description("Number of sweeps through the entire training set."))
  WorkflowProperty(TrialEpochCount)
  WorkflowProperty(BatchSize, Description("Number of images used per gradient update."))
  WorkflowProperty(FilterBatchSize, Description("Number of filters that are processed in parallel."))
  WorkflowProperty(GpuCount, Description("Number of GPUs used for training."))

  WorkflowProperty(SparsityMethod, Enumerator<Type>())
  WorkflowProperty(SparsityTarget, Description("Target expected activation of a hidden unit."))
  WorkflowProperty(SparsityWeight, Description("Weight of the sparsity target relative to the learning rate."))

  WorkflowProperty(CdIterations, Description("Number of CD iterations. (1: CD learning, >1: CD-n learning)"))
  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(LearningRates, NotEmpty<Type>(), Description("Initial values of the decaying learning rates of the filters."))
  WorkflowProperty(LearningDecay, Description("After how many epochs the learning rate will be halved. A value of -1 indicates no LearningDecay."))
  WorkflowProperty(InitialMomentum, Description("Momentum used for the first epoch."))
  WorkflowProperty(FinalMomentum)
  WorkflowProperty(MomentumDecayEpochs)
  WorkflowProperty(WeightDecay)
  WorkflowProperty(InitialWeights, Description("If given, these weights will be tested as initial weights and will override the initial weights."))
  WorkflowProperty(SignalToNoiseRatio, Description("If greater than 0, the standard deviation of the noise of the initial filters is set to 'sd(learned filters) / SNR'."))
  WorkflowProperty(RandomizeTraining, Flag(), Description("Randomly select images of a mini-batch."))
  WorkflowProperty(ShareBiasTerms, Flag(), Description("If checked, visible and hidden units of the same filter share bias terms."))
  WorkflowProperty(ChannelsPerBlock, Description("All channels of the same pooling block share the same bias terms when shared bias terms are active. Hence, this number must be known."))
  WorkflowProperty(DropoutMethod, Enumerator<Type>(), Description("Defines if entire columns or individual hidden units are dropped."))
  WorkflowProperty(VisibleDropout, Description("Probability of a visible unit of being ignored. (currently not used)"))
  WorkflowProperty(HiddenDropout, Description("Probability of a hidden unit of being ignored."))
  WorkflowProperty(FilterDropout, Description("Probability of an entire batch of filters being ignored. To drop individual filters, set the filter batch size to 1"))
  WorkflowProperty(CalculateError, Flag(), Description("If checked, the reconstruction error is calculated"))
  WorkflowProperty(UpdateModel, Description("If greater than 0, the model is updated every <UpdateModel> epochs."))


  WorkflowProperty(CurrentEpoch, NoParameter())
  WorkflowProperty(Model, Output("CRBM"), Description("The trained model."))
  WorkflowProperty(AverageEpochTime, Output("T"))
  WorkflowProperty(ReconstructionError, NoParameter())

EndPropertyDefinitions

Trainer::Trainer()
 : _EpochCount(100), _TrialEpochCount(10), _BatchSize(20), _FilterBatchSize(1), _GpuCount(1),
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
