/*
 * StackTensors.h
 *
 *  Created on: Jun 24, 2013
 *      Author: tombr
 */

#ifndef GML_STACKTENSORS_H_
#define GML_STACKTENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

/**
 * Creates a multi-channel tensor by stacking two tensors together
 */
class StackTensors : public DefaultWorkflowElement<StackTensors> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(StackTensors)

  Property(Tensor1, boost::shared_ptr<tensor_t>)
  Property(Tensor2, boost::shared_ptr<tensor_t>)
  Property(OutputTensor, boost::shared_ptr<tensor_t>)

public:
  StackTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_STACKTENSORS_H_ */
