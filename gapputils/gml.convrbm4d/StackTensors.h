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

#include <capputils/Enumerators.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

CapputilsEnumerator(StackMode, SingleTensor, TensorVector);

/**
 * Creates a multi-channel tensor by stacking two tensors together
 */
class StackTensors : public DefaultWorkflowElement<StackTensors> {

  typedef model_t::host_tensor_t tensor_t;
  typedef model_t::v_host_tensor_t v_tensor_t;

  InitReflectableClass(StackTensors)

  Property(InputTensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Mode, StackMode)
  Property(OutputTensor, boost::shared_ptr<tensor_t>)
  Property(OutputTensors, boost::shared_ptr<v_tensor_t>)

public:
  StackTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_STACKTENSORS_H_ */
