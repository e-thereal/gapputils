/*
 * TensorWriter.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GML_CONVRBM4D_TENSORWRITER_H_
#define GML_CONVRBM4D_TENSORWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class TensorWriter : public DefaultWorkflowElement<TensorWriter> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(TensorWriter)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Filename, std::string)
  Property(OutputName, std::string)

public:
  TensorWriter();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_CONVRBM4D_TENSORWRITER_H_ */
