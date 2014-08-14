/*
 * FineTuning.h
 *
 *  Created on: Jul 21, 2014
 *      Author: tombr
 */

#ifndef GML_FINETUNING_H_
#define GML_FINETUNING_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbn {

struct FineTuningChecker { FineTuningChecker(); };

class FineTuning : public DefaultWorkflowElement<FineTuning> {

  typedef crbm_t::host_tensor_t host_tensor_t;
  typedef crbm_t::v_host_tensor_t v_host_tensor_t;

  friend class FineTuningChecker;

  InitReflectableClass(FineTuning)

  Property(InitialModel, boost::shared_ptr<dbn_t>)
  Property(Tensors, boost::shared_ptr<v_host_tensor_t>)

  Property(EpochCount, int)
  Property(BatchSize, int)
  Property(GpuCount, int)
  Property(FilterBatchLength, std::vector<int>)

  Property(LearningRate, double)
  Property(InitialMomentum, double)
  Property(FinalMomentum, double)
  Property(MomentumDecayEpochs, int)
  Property(WeightDecay, double)
  Property(RandomizeTraining, bool)

  Property(Model, boost::shared_ptr<dbn_t>)
  Property(ReconstructionError, double)

public:
  FineTuning();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */

} /* namespace gml */

#endif /* GML_FINETUNING_H_ */
