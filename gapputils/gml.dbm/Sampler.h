/*
 * Sampler.h
 *
 *  Created on: Jul 15, 2013
 *      Author: tombr
 */

#ifndef GML_SAMPLER_H_
#define GML_SAMPLER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace dbm {

struct SamplerChecker { SamplerChecker(); };

class Sampler : public DefaultWorkflowElement<Sampler> {

  typedef Model::tensor_t host_tensor_t;
  typedef Model::v_tensor_t v_host_tensor_t;
  typedef Model::vv_tensor_t vv_host_tensor_t;

  friend class SamplerChecker;

  InitReflectableClass(Sampler)

  Property(Model, boost::shared_ptr<Model>)
  Property(GpuCount, int)
  Property(SampleCount, int)
  Property(Iterations, int)
  Property(Damped, bool)
  Property(Samples, boost::shared_ptr<v_host_tensor_t>)

public:
  Sampler();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_SAMPLER_H_ */
