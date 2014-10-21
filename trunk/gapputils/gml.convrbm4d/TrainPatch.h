/*
 * TrainPatch.h
 *
 *  Created on: Oct 16, 2014
 *      Author: tombr
 */

#ifndef GML_TRAINPATCH_H_
#define GML_TRAINPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include "Model.h"

#include <tbblas/deeplearn/sparsity_method.hpp>
#include <tbblas/deeplearn/dropout_method.hpp>

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(DbmLayer, RBM, VisibleLayer, IntermediateLayer, TopLayer);

struct TrainPatchChecker { TrainPatchChecker(); };

class TrainPatch : public DefaultWorkflowElement<TrainPatch> {
public:
  typedef model_t::value_t value_t;
  typedef model_t::host_tensor_t host_tensor_t;
  typedef model_t::v_host_tensor_t v_host_tensor_t;

  friend class TrainPatchChecker;

  InitReflectableClass(TrainPatch)

  // Primary inputs
  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(Tensors, boost::shared_ptr<v_host_tensor_t>)
  Property(DbmLayer, DbmLayer)

  // Data set size dependent parameters
  Property(PatchCount, int)
  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(FilterBatchSize, int)
  Property(GpuCount, int)

  // Sparsity parameters
  Property(SparsityMethod, tbblas::deeplearn::sparsity_method)
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
  Property(ChannelsPerBlock, int)
  Property(DropoutMethod, tbblas::deeplearn::dropout_method)
  Property(VisibleDropout, double)
  Property(HiddenDropout, double)
  Property(FilterDropout, double)
  Property(CalculateError, bool)
  Property(UpdateModel, int)
  int dummy3;

  // Output parameters
  Property(CurrentEpoch, int)
  Property(Model, boost::shared_ptr<model_t>)
  Property(AverageEpochTime, double)
  Property(ReconstructionError, double)
  Property(Patches, boost::shared_ptr<v_host_tensor_t>)

public:
  TrainPatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_TRAINER_H_ */
