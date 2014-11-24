/*
 * ManipulateTensor.h
 *
 *  Created on: Nov 10, 2014
 *      Author: tombr
 */

#ifndef GML_MANIPULATETENSOR_H_
#define GML_MANIPULATETENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class ManipulateTensor : public DefaultWorkflowElement<ManipulateTensor> {

  InitReflectableClass(ManipulateTensor)

  Property(Input, boost::shared_ptr<host_tensor_t>)
  Property(Mask, std::vector<int>)
  Property(Output, boost::shared_ptr<host_tensor_t>)

public:
  ManipulateTensor();

protected:
  void update(IProgressMonitor* monitor) const;

};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_MANIPULATETENSOR_H_ */
