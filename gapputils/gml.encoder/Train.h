/*
 * Train.h
 *
 *  Created on: Jan 06, 2015
 *      Author: tombr
 */

#ifndef GML_TRAIN_H_
#define GML_TRAIN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

#include <tbblas/deeplearn/objective_function.hpp>

namespace gml {

namespace encoder {

CapputilsEnumerator(TrainingMethod, Momentum, AdaDelta, Adam, AdamDecay)

struct TrainChecker { TrainChecker(); } ;

class Train : public DefaultWorkflowElement<Train> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  friend class TrainChecker;

  InitReflectableClass(Train)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(EpochCount, int)
  Property(TrialEpochCount, int)
  Property(BatchSize, int)
  Property(FilterBatchSize, std::vector<int>)
  Property(SubRegionCount, host_tensor_t::dim_t)
  Property(Objective, tbblas::deeplearn::objective_function)
  Property(SensitivityRatio, double)
  Property(SharedBiasTerms, bool)

  Property(Method, TrainingMethod)
  Property(LearningRates, std::vector<double>)
  Property(LearningDecay, int)
  Property(WeightCosts, double)
  Property(InitialWeights, std::vector<double>)
  Property(RandomizeTraining, bool)

  Property(AugmentedChannels, std::vector<int>)
  Property(ContrastSd, double)
  Property(BrightnessSd, double)
  Property(GammaSd, double)

  Property(BestOfN, int)
  Property(SaveEvery, int)

  Property(CurrentEpoch, int)
  Property(Model, boost::shared_ptr<model_t>)
  Property(Model2, boost::shared_ptr<model_t>)
  Property(BestModel, boost::shared_ptr<model_t>)
  Property(WorstModel, boost::shared_ptr<model_t>)
  Property(AugmentedSet, boost::shared_ptr<v_host_tensor_t>)

public:
  Train();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace encoder */

} /* namespace gml */

#endif /* GML_TRAIN_H_ */
