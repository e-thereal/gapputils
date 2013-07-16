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

#include "DbmModel.h"

namespace gml {

namespace convrbm4d {

struct SamplerChecker { SamplerChecker(); };

class Sampler : public DefaultWorkflowElement<Sampler> {

  typedef DbmModel::tensor_t host_tensor_t;
  typedef DbmModel::v_tensor_t v_host_tensor_t;
  typedef DbmModel::vv_tensor_t vv_host_tensor_t;

  friend class SamplerChecker;

  InitReflectableClass(Sampler)

  Property(Model, boost::shared_ptr<DbmModel>)
  Property(SampleCount, int)
  Property(Iterations, int)
  Property(GpuCount, int)
  Property(Samples, boost::shared_ptr<v_host_tensor_t>)

public:
  Sampler();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_SAMPLER_H_ */
