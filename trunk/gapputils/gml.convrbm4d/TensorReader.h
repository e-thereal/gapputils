/*
 * TensorReader.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GML_CONVRBM4D_TENSORREADER_H_
#define GML_CONVRBM4D_TENSORREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class TensorReader : public DefaultWorkflowElement<TensorReader> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(TensorReader)

  Property(Filename, std::string)
  Property(FirstIndex, int)
  Property(MaxCount, int)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(Channels, int)
  Property(TensorCount, int)

public:
  TensorReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_CONVRBM4D_TENSORREADER_H_ */
