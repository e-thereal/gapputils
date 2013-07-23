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

#include "Model.h"
#include "CodingDirection.h"

namespace gml {

namespace dbm {

struct InferenceChecker { InferenceChecker(); };

class Inference : public DefaultWorkflowElement<Inference> {

  typedef Model::value_t value_t;
  typedef Model::tensor_t host_tensor_t;
  typedef Model::v_tensor_t v_host_tensor_t;
  typedef Model::vv_tensor_t vv_host_tensor_t;
  typedef Model::matrix_t host_matrix_t;
  typedef Model::v_matrix_t v_host_matrix_t;

  friend class InferenceChecker;

  InitReflectableClass(Inference)

  Property(Model, boost::shared_ptr<Model>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(Direction, CodingDirection)
  Property(Iterations, int)
  Property(GpuCount, int)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  Inference();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace dbm */

} /* namespace gml */

#endif /* GML_INFERENCE_H_ */
