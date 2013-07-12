/*
 * Inference.h
 *
 *  Created on: Jul 12, 2013
 *      Author: tombr
 */

#ifndef GML_INFERENCE_H_
#define GML_INFERENCE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "DbmModel.h"

namespace gml {

namespace convrbm4d {

struct InferenceChecker { InferenceChecker(); };

class Inference : public DefaultWorkflowElement<Inference> {

  typedef DbmModel::value_t value_t;
  typedef DbmModel::tensor_t host_tensor_t;
  typedef DbmModel::v_tensor_t v_host_tensor_t;
  typedef DbmModel::vv_tensor_t vv_host_tensor_t;

  friend class InferenceChecker;

  InitReflectableClass(Inference)

  Property(Model, boost::shared_ptr<DbmModel>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(GpuCount, int)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  Inference();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_INFERENCE_H_ */
