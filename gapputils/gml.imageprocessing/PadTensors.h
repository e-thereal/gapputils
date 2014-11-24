/*
 * PadTensors.h
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#ifndef GML_PADTENSORS_H_
#define GML_PADTENSORS_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class PadTensors : public DefaultWorkflowElement<PadTensors> {

  typedef gapputils::host_tensor_t tensor_t;
  typedef gapputils::v_host_tensor_t v_tensor_t;

  InitReflectableClass(PadTensors)

  Property(InputTensor, boost::shared_ptr<tensor_t>)
  Property(InputTensors, boost::shared_ptr<v_tensor_t>)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(ReversePadding, bool)
  Property(OutputTensor, boost::shared_ptr<tensor_t>)
  Property(OutputTensors, boost::shared_ptr<v_tensor_t>)

public:
  PadTensors();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_PADTENSORS_H_ */
