/*
 * FineTuning.h
 *
 *  Created on: Jul 16, 2013
 *      Author: tombr
 */

#ifndef GML_FINETUNING_H_
#define GML_FINETUNING_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbm {

struct FineTuningChecker { FineTuningChecker(); } ;

class FineTuning : public DefaultWorkflowElement<FineTuning> {

  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;
  typedef Model::v_tensor_t v_host_tensor_t;
  typedef Model::vv_tensor_t vv_host_tensor_t;

  typedef Model::matrix_t host_matrix_t;
  typedef Model::v_matrix_t v_host_matrix_t;

  friend class FineTuningChecker;

  InitReflectableClass(FineTuning)

  Property(Dataset, boost::shared_ptr<v_host_tensor_t>)
  Property(InitialModel, boost::shared_ptr<Model>)
  Property(GpuCount, int)
  Property(EpochCount, int)
  Property(BatchSize, int)
  int dummy;
  Property(LearningRate, double)
  Property(LearningDecay, int)
  Property(MeanFieldIterations, int)
  Property(GibbsIterations, int)
  Property(SampleCount, int)
  Property(OutputModel, boost::shared_ptr<Model>)

public:
  FineTuning();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_FINETUNING_H_ */
