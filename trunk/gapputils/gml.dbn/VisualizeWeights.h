/*
 * VisualizeWeights.h
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#ifndef GML_VISUALIZEWEIGHTS_H_
#define GML_VISUALIZEWEIGHTS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbn {

struct VisualizeWeightsChecker { VisualizeWeightsChecker(); };

class VisualizeWeights : public DefaultWorkflowElement<VisualizeWeights> {

  typedef crbm_t::host_tensor_t host_tensor_t;
  typedef crbm_t::v_host_tensor_t v_host_tensor_t;

  friend class VisualizeWeightsChecker;

  InitReflectableClass(VisualizeWeights)

  Property(Model, boost::shared_ptr<dbn_t>)
  Property(FilterBatchLength, std::vector<int>)
  Property(Weights, boost::shared_ptr<v_host_tensor_t>)

public:
  VisualizeWeights();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbn */

} /* namespace gml */

#endif /* VISUALIZEWEIGHTS_H_ */
