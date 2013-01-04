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
namespace convrbm4d {

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
  Property(GpuCount, int)
  int dummy;
  Property(LearningRateW, double)
  Property(LearningRateVB, double)
  Property(LearningRateHB, double)
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)
  Property(RandomizeTraining, bool)
  Property(CalculateError, bool)
  Property(ShareBiasTerms, bool)
  Property(Logfile, std::string)
  Property(MonitorEvery, int)
  Property(ReconstructionCount, int)

  Property(Model, boost::shared_ptr<Model>)
  Property(Filters, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(VisibleBiases, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(HiddenBiases, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(HiddenUnits, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)
  Property(Reconstructions, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)

public:
  Trainer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* TRAINER_H_ */
