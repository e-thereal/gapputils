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

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace core {

CapputilsEnumerator(StackMode, SingleTensor, TensorVector);

/**
 * Creates a multi-channel tensor by stacking two tensors together
 */
class StackTensors : public DefaultWorkflowElement<StackTensors> {

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

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

}

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_STACKTENSORS_H_ */
