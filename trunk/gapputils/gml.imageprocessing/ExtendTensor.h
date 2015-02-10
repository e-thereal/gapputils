/*
 * ExtendTensor.h
 *
 *  Created on: Feb 05, 2015
 *      Author: tombr
 */

#ifndef GML_EXTENDTENSOR_H_
#define GML_EXTENDTENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class ExtendTensor : public DefaultWorkflowElement<ExtendTensor> {

  InitReflectableClass(ExtendTensor)

  Property(Input, boost::shared_ptr<host_tensor_t>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(WidthFactor, int)
  Property(HeightFactor, int)
  Property(DepthFactor, int)
  Property(Output, boost::shared_ptr<host_tensor_t>)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  ExtendTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_EXTENDTENSOR_H_ */
