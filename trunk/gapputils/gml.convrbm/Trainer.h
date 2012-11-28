/*
 * Trainer.h
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#ifndef TRAINER_H_
#define TRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {
namespace convrbm {

struct TrainerChecker { TrainerChecker(); };

class Trainer : public DefaultWorkflowElement<Trainer> {
public:
  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;

  friend class TrainerChecker;

  InitReflectableClass(Trainer)

  Property(InitialModel, boost::shared_ptr<Model>)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(LearningRate, double)
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)

  Property(Model, boost::shared_ptr<Model>)
  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)

public:
  Trainer();
  virtual ~Trainer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm */
} /* namespace gml */
#endif /* TRAINER_H_ */
