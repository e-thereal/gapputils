/*
 * ResampleTensor.h
 *
 *  Created on: Feb 5, 2015
 *      Author: tombr
 */

#ifndef GML_RESAMPLETENSOR_H_
#define GML_RESAMPLETENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

struct ResampleTensorChecker { ResampleTensorChecker(); };

class ResampleTensor : public DefaultWorkflowElement<ResampleTensor> {

  InitReflectableClass(ResampleTensor)

  friend class ResampleTensorChecker;

  Property(Input, boost::shared_ptr<host_tensor_t>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(Size, host_tensor_t::dim_t)
  Property(Output, boost::shared_ptr<host_tensor_t>)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  ResampleTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_RESAMPLETENSOR_H_ */
