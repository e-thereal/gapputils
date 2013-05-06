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

#include "SparsityMethod.h"
#include "DropoutMethod.h"

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

  // Primary inputs
  Property(InitialModel, boost::shared_ptr<Model>)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<host_tensor_t> > >)

  // Data set size dependent parameters
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(GpuCount, int)
  int dummy;

  // Sparsity parameters
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)
  Property(SparsityMethod, SparsityMethod)

  // Learning algorithm parameters
  Property(LearningRate, double)
  Property(LearningDecay, double)
  Property(InitialMomentum, double)
  Property(FinalMomentum, double)
  Property(MomentumDecayEpochs, int)
  Property(WeightDecay, double)
  Property(WeightVectorLimit, double)
  Property(RandomizeTraining, bool)
  Property(ShareBiasTerms, bool)
  Property(VisibleDropout, double)
  Property(HiddenDropout, double)
  Property(FilterDropout, double)
  Property(DropoutMethod, DropoutMethod)
  Property(DropoutStage, DropoutStage)
  Property(CalculateError, bool)
  Property(UpdateModel, int)
  int dummy2;

  // Output parameters
  Property(CurrentEpoch, int)
  Property(Model, boost::shared_ptr<Model>)
  Property(ModelIncrement, boost::shared_ptr<Model>)
  Property(AverageEpochTime, double)
  Property(ReconstructionError, double)

public:
  Trainer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* TRAINER_H_ */
