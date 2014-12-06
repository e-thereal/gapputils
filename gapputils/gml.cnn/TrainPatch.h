/*
 * TrainPatch.h
 *
 *  Created on: Dec 01, 2014
 *      Author: tombr
 */

#ifndef GML_TRAINPATCH_H_
#define GML_TRAINPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace cnn {

CapputilsEnumerator(TrainingMethod, Momentum, AdaDelta)

struct TrainPatchChecker { TrainPatchChecker(); } ;

class TrainPatch : public DefaultWorkflowElement<TrainPatch> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  friend class TrainPatchChecker;

  InitReflectableClass(TrainPatch)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(Mask, boost::shared_ptr<host_tensor_t>)
  Property(EpochCount, int)
  Property(TrialEpochCount, int)
  Property(BatchSize, int)
  Property(FilterBatchSize, std::vector<int>)
  Property(PatchCounts, std::vector<int>)
  Property(MultiPatchCount, int)
  Property(PositiveRatio, double)

  Property(Method, TrainingMethod)
  Property(LearningRates, std::vector<double>)
  Property(LearningDecay, int)
  Property(WeightCosts, double)
  Property(InitialWeights, std::vector<double>)
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

} /* namespace cnn */

} /* namespace gml */

#endif /* GML_TRAINPATCH_H_ */
