/*
 * PredictPatch.h
 *
 *  Created on: Dec 06, 2014
 *      Author: tombr
 */

#ifndef GML_PREDICTPATCH_H_
#define GML_PREDICTPATCH_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>
#include <gapputils/Tensor.h>

#include "Model.h"

namespace gml {

namespace nn {

struct PredictPatchChecker { PredictPatchChecker(); };

class PredictPatch : public DefaultWorkflowElement<PredictPatch> {

  friend class PredictPatchChecker;

  InitReflectableClass(PredictPatch)

  Property(Model, boost::shared_ptr<model_t>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(PatchWidth, int)
  Property(PatchHeight, int)
  Property(PatchDepth, int)
  Property(PatchCounts, std::vector<int>)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  PredictPatch();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace nn */

} /* namespace gml */

#endif /* GML_PREDICTPATCH_H_ */
