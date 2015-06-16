/*
 * OpenTensor.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GML_OPENTENSOR_H_
#define GML_OPENTENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace io {

class OpenTensor : public DefaultWorkflowElement<OpenTensor> {

  typedef tbblas::tensor<float, 4> tensor_t;

  InitReflectableClass(OpenTensor)

  Property(Filename, std::string)
  Property(SingleTensor, bool)
  Property(FirstIndex, int)
  Property(MaxCount, int)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(Channels, int)
  Property(TensorCount, int)

public:
  OpenTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* io */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_OPENTENSOR_H_ */
