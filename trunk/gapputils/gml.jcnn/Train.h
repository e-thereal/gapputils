/*
 * Train.h
 *
 *  Created on: Nov 24, 2014
 *      Author: tombr
 */

#ifndef GML_TRAIN_H_
#define GML_TRAIN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace jcnn {

CapputilsEnumerator(TrainingMethod, Momentum, AdaDelta)

struct TrainChecker { TrainChecker(); } ;

class Train : public DefaultWorkflowElement<Train> {

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  friend class TrainChecker;

  InitReflectableClass(Train)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(LeftTrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(RightTrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(LeftFilterBatchSize, std::vector<int>)
  Property(RightFilterBatchSize, std::vector<int>)

  Property(Method, TrainingMethod)
  Property(CLearningRate, double)
  Property(DLearningRate, double)
  Property(WeightCosts, double)
  Property(RandomizeTraining, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  Train();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace jcnn */

} /* namespace gml */

#endif /* GML_TRAIN_H_ */
