/*
 * Train.h
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#ifndef GML_TRAIN_H_
#define GML_TRAIN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace nn {

struct TrainChecker { TrainChecker(); } ;

class Train : public DefaultWorkflowElement<Train> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  friend class TrainChecker;

  InitReflectableClass(Train)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_data_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(EpochCount, int)
  Property(BatchSize, int)

  Property(LearningRate, double)
  Property(WeightCosts, double)
  Property(ShuffleTrainingSet, bool)
  Property(Model, boost::shared_ptr<model_t>)

public:
  Train();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_TRAIN_H_ */
