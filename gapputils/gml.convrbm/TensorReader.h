/*
 * TensorReader.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GML_CONVRBM_TENSORREADER_H_
#define GML_CONVRBM_TENSORREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm {

class TensorReader : public DefaultWorkflowElement<TensorReader> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(TensorReader)

  Property(Filename, std::string)
  Property(MaxCount, int)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Width, int)
  Property(Height, int)
  Property(FilterCount, int)
  Property(TensorCount, int)

public:
  TensorReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm */

} /* namespace gml */

#endif /* GML_CONVRBM_TENSORREADER_H_ */
