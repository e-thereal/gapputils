/*
 * TensorRange.h
 *
 *  Created on: May 21, 2015
 *      Author: tombr
 */

#ifndef GML_TENSORRANGE_H_
#define GML_TENSORRANGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class TensorRange : public DefaultWorkflowElement<TensorRange> {

  InitReflectableClass(TensorRange)

  Property(Inputs, boost::shared_ptr<v_host_tensor_t>)
  Property(First, int)
  Property(Count, int)
  Property(Outputs, boost::shared_ptr<v_host_tensor_t>)

public:
  TensorRange();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_TENSORRANGE_H_ */
