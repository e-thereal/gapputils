/*
 * Trainer.h
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

#ifndef GML_TRAINER_H_
#define GML_TRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "SparsityMethod.h"
#include "DropoutMethod.h"

#include "Model.h"

namespace gml {
namespace convrbm4d {

CapputilsEnumerator(DbmLayer, RBM, VisibleLayer, IntermediateLayer, TopLayer);

struct TrainerChecker { TrainerChecker(); };

class Trainer : public DefaultWorkflowElement<Trainer> {
public:
  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  friend class TrainerChecker;

  InitReflectableClass(Trainer)

  // Primary inputs
  Property(InitialModel, boost::shared_ptr<Model>)
  Property(Tensors, boost::shared_ptr<v_host_tensor_t>)
  Property(DbmLayer, DbmLayer)

  // Data set size dependent parameters
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(FilterBatchSize, int)
  Property(GpuCount, int)
  Property(PadInputs, bool)

  // Sparsity parameters
  Property(SparsityMethod, SparsityMethod)
  int dummy1;
  Property(SparsityTarget, double)
  Property(SparsityWeight, double)

  // Learning algorithm parameters
  Property(CdIterations, int)
  int dummy4;
  Property(LearningRate, double)
  Property(LearningDecay, double)
  Property(InitialMomentum, double)
  Property(FinalMomentum, double)
  Property(MomentumDecayEpochs, int)
  int dummy2;
  Property(WeightDecay, double)
  Property(WeightVectorLimit, double)
  Property(RandomizeTraining, bool)
  Property(ShareBiasTerms, bool)
  Property(ChannelsPerBlock, int)
  Property(DropoutMethod, DropoutMethod)
  Property(DropoutStage, DropoutStage)
  Property(VisibleDropout, double)
  Property(HiddenDropout, double)
  Property(FilterDropout, double)
  Property(CalculateError, bool)
  Property(UpdateModel, int)
  int dummy3;

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

#endif /* GML_TRAINER_H_ */
