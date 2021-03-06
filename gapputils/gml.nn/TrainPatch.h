/*
 * TrainPatch.h
 *
 *  Created on: Oct 14, 2014
 *      Author: tombr
 */

#ifndef GML_TRAINPATCH_H_
#define GML_TRAINPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"
#include "TrainingMethod.h"

#include <tbblas/deeplearn/objective_function.hpp>

namespace gml {

namespace nn {

CapputilsEnumerator(PatchSelectionMethod, PositiveRatio, LeitnerSystem)

struct TrainPatchChecker { TrainPatchChecker(); } ;

class TrainPatch : public DefaultWorkflowElement<TrainPatch> {

  typedef tbblas::tensor<float, 4> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  friend class TrainPatchChecker;

  InitReflectableClass(TrainPatch)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(Mask, boost::shared_ptr<host_tensor_t>)
  Property(PatchWidth, int)
  Property(PatchHeight, int)
  Property(PatchDepth, int)
  Property(PatchCount, int)
  Property(SelectionMethod, PatchSelectionMethod)
  Property(PositiveRatio, double)
  Property(MinimumBucketSizes, std::vector<int>)
  Property(BucketRatio, double)

  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(Objective, tbblas::deeplearn::objective_function)
  Property(SensitivityRatio, double)
  Property(Method, TrainingMethod)
  Property(LearningRate, double)
  Property(WeightCosts, double)
  Property(DropoutRates, std::vector<double>)
  Property(RandomizeTraining, bool)
  Property(Model, boost::shared_ptr<model_t>)
  Property(Patches, boost::shared_ptr<v_host_tensor_t>)
  Property(Targets, boost::shared_ptr<v_host_tensor_t>)
  Property(Predictions, boost::shared_ptr<v_host_tensor_t>)

public:
  TrainPatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_TRAINPATCH_H_ */
