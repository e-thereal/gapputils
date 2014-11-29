/*
 * Jaccobian.h
 *
 *  Created on: Nov 26, 2014
 *      Author: tombr
 */

#ifndef GML_JACCOBIAN_H_
#define GML_JACCOBIAN_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class Jaccobian : public DefaultWorkflowElement<Jaccobian> {

  InitReflectableClass(Jaccobian)

  Property(Input, boost::shared_ptr<host_tensor_t>)
  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(Output, boost::shared_ptr<host_tensor_t>)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  Jaccobian();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_JACCOBIAN_H_ */
