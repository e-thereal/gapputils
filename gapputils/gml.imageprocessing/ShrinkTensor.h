/*
 * ShrinkTensor.h
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#ifndef GML_SHRINKTENSOR_H_
#define GML_SHRINKTENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(ShrinkingMethod, Average, Maximum, Minimum);

class ShrinkTensor : public DefaultWorkflowElement<ShrinkTensor> {

  InitReflectableClass(ShrinkTensor)

  Property(InputTensor, boost::shared_ptr<host_tensor_t>)
  Property(WidthFactor, int)
  Property(HeightFactor, int)
  Property(DepthFactor, int)
  Property(ShrinkingMethod, ShrinkingMethod)
  Property(OutputTensor, boost::shared_ptr<host_tensor_t>)

public:
  ShrinkTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_SHRINKTENSOR_H_ */
