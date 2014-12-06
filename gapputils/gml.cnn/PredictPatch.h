/*
 * PredictPatch.h
 *
 *  Created on: Dec 04, 2014
 *      Author: tombr
 */

#ifndef GML_PREDICTPATCH_H_
#define GML_PREDICTPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace cnn {

struct PredictPatchChecker { PredictPatchChecker(); };

class PredictPatch : public DefaultWorkflowElement<PredictPatch> {

  friend class PredictPatchChecker;

  typedef model_t::value_t value_t;
  static const unsigned dimCount = model_t::dimCount;

  typedef tbblas::tensor<value_t, dimCount> host_tensor_t;
  typedef std::vector<boost::shared_ptr<host_tensor_t> > v_host_tensor_t;

  InitReflectableClass(PredictPatch)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)

  Property(FilterBatchSize, std::vector<int>)
  Property(PatchCounts, std::vector<int>)

  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  PredictPatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cnn */

} /* namespace gml */

#endif /* GML_PREDICTPATCH_H_ */
