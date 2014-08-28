/*
 * SaveTensor.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GML_SAVETENSOR_H_
#define GML_SAVETENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace io {

class SaveTensor : public DefaultWorkflowElement<SaveTensor> {

  typedef tbblas::tensor<float, 4> tensor_t;

  InitReflectableClass(SaveTensor)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  SaveTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* io */

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_SAVETENSOR_H_ */
