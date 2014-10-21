/*
 * TrainPatch.cpp
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#include "TrainPatch.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(TrainPatch, Description("Trains a convolutional RBM using CD-n learning on selected patches of the training set. Can also be used for layer-wise pretraining of DBMs (see the DbmLayer parameter)."))

  ReflectableBase(DefaultWorkflowElement<TrainPatch>)

  WorkflowProperty(InitialModel, Input("CRBM"), NotNull<Type>(), Description("Required initial model. The initial model can be a previously trained model, are a new model initialized using Initialize"))
  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>(), Description("The training set."))
  WorkflowProperty(DbmLayer, Enumerator<Type>())

  WorkflowProperty(PatchCount, Description("Number of a training patches per image."))

  WorkflowProperty(EpochCount, Description("Number of sweeps through the entire training set."))
  WorkflowProperty(BatchSize, Description("Number of images used per gradient update."))
  WorkflowProperty(FilterBatchSize, Description("Number of filters that are processed in parallel."))
  WorkflowProperty(GpuCount, Description("Number of GPUs used for training."))

  WorkflowProperty(SparsityMethod, Enumerator<Type>())
  WorkflowProperty(SparsityTarget, Description("Target expected activation of a hidden unit."))
  WorkflowProperty(SparsityWeight, Description("Weight of the sparsity target relative to the learning rate."))

  WorkflowProperty(CdIterations, Description("Number of CD iterations. (1: CD learning, >1: CD-n learning)"))
  WorkflowProperty(LearningRate, Description("Initial value of the exponentially decaying learning rate."))
  WorkflowProperty(LearningDecay, Description("Rate at which the learning rate decays."))
  WorkflowProperty(InitialMomentum, Description("Momentum used for the first epoch."))
  WorkflowProperty(FinalMomentum)
  WorkflowProperty(MomentumDecayEpochs)
  WorkflowProperty(WeightDecay)
  WorkflowProperty(WeightVectorLimit, Description("Maximum length of the weight update vector. Lower values reduce oscillation."))
  WorkflowProperty(RandomizeTraining, Flag(), Description("Randomly select images of a mini-batch."))
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
  WorkflowProperty(Patches, Output("Ps"))

EndPropertyDefinitions

TrainPatch::TrainPatch()
 : _PatchCount(10), _EpochCount(100), _BatchSize(20), _FilterBatchSize(1), _GpuCount(1),
   _SparsityTarget(1e-2), _SparsityWeight(0.1),
   _CdIterations(1), _LearningRate(1e-3), _LearningDecay(0.98), _InitialMomentum(0.5), _FinalMomentum(0.9),
   _MomentumDecayEpochs(20), _WeightDecay(0), _WeightVectorLimit(1), _RandomizeTraining(false),
   _ChannelsPerBlock(1), _VisibleDropout(0.0), _HiddenDropout(0.5), _FilterDropout(0.0),
   _CalculateError(false), _UpdateModel(0),
   _CurrentEpoch(0), _AverageEpochTime(0.0), _ReconstructionError(0.0)
{
  setLabel("TrainPatch");
}

TrainPatchChecker trainPatchChecker;

} /* namespace convrbm */

} /* namespace gml */
