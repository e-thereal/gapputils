/*
 * FindThreshold.h
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#ifndef GML_FINDTHRESHOLD_H_
#define GML_FINDTHRESHOLD_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include "Model.h"

namespace gml {

namespace nn {

struct FindThresholdChecker { FindThresholdChecker(); };

class FindThreshold : public DefaultWorkflowElement<FindThreshold> {

  static const int dimCount = host_tensor_t::dimCount;
  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;

  friend class FindThresholdChecker;

  InitReflectableClass(FindThreshold)

  Property(InitialModel, boost::shared_ptr<model_t>)
  Property(TrainingSet, boost::shared_ptr<v_host_tensor_t>)
  Property(Labels, boost::shared_ptr<v_host_tensor_t>)
  Property(PatchWidth, int)
  Property(PatchHeight, int)
  Property(PatchDepth, int)
  Property(PatchCounts, std::vector<int>)
  Property(Model, boost::shared_ptr<patch_model_t>)

private:
  mutable std::vector<host_tensor_t> predictions;

public:
  FindThreshold();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_FINDTHRESHOLD_H_ */
