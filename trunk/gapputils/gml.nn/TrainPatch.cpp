/*
 * TrainPatch.cpp
 *
 *  Created on: Oct 14, 2014
 *      Author: tombr
 */

#include "TrainPatch.h"

namespace gml {

namespace nn {


BeginPropertyDefinitions(TrainPatch)

  ReflectableBase(DefaultWorkflowElement<TrainPatch>)

  WorkflowProperty(InitialModel, Input("NN"), NotNull<Type>())
  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mask, Input("M"))
  WorkflowProperty(PatchWidth)
  WorkflowProperty(PatchHeight)
  WorkflowProperty(PatchDepth)
  WorkflowProperty(PatchCount)

  WorkflowProperty(SelectionMethod, Enumerator<Type>())
  WorkflowProperty(PositiveRatio)
  WorkflowProperty(MinimumBucketSizes, Description("The first bucket will always be refilled with random samples (balanced by the positive ratio) to the minimum size. Samples from other buckets are only drawn, if they contain at least the minimum number of samples."))
  WorkflowProperty(BucketRatio, Description("Ratio of choosing the current bucket over a higher bucket."))

  WorkflowProperty(EpochCount)
  WorkflowProperty(BatchSize)
  WorkflowProperty(Objective, Enumerator<Type>())
  WorkflowProperty(SensitivityRatio)
  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(LearningRate)
  WorkflowProperty(WeightCosts)
  WorkflowProperty(DropoutRates)
  WorkflowProperty(RandomizeTraining, Flag())
  WorkflowProperty(Model, Output("NN"))
  WorkflowProperty(Patches, Output("Ps"))
  WorkflowProperty(Targets, Output("Ts"))
  WorkflowProperty(Predictions, Output("Ps"))

EndPropertyDefinitions

TrainPatch::TrainPatch()
  : _PatchWidth(16), _PatchHeight(16), _PatchDepth(16), _PatchCount(16),
    _PositiveRatio(0.5), _BucketRatio(0.5), _EpochCount(100), _BatchSize(50),
    _SensitivityRatio(0.5), _LearningRate(0.0001), _WeightCosts(0.0002), _RandomizeTraining(true)
{
  setLabel("TrainPatch");
  _MinimumBucketSizes.push_back(100);
  _MinimumBucketSizes.push_back(100);
  _MinimumBucketSizes.push_back(100);
  _MinimumBucketSizes.push_back(100);
}

TrainPatchChecker trainPatchChecker;

} /* namespace nn */

} /* namespace gml */
