/*
 * CorrelationTensor.h
 *
 *  Created on: Dec 10, 2013
 *      Author: tombr
 */

#ifndef GML_CORRELATIONTENSOR_H_
#define GML_CORRELATIONTENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class CorrelationTensor : public DefaultWorkflowElement<CorrelationTensor> {

  typedef Model::value_t value_t;
  typedef Model::tensor_t tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef std::vector<double> data_t;

  InitReflectableClass(CorrelationTensor)

  Property(Tensors, boost::shared_ptr<v_tensor_t>)
  Property(Data, boost::shared_ptr<data_t>)
  Property(CorrelationTensor, boost::shared_ptr<tensor_t>)

public:
  CorrelationTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */

} /* namespace gml */

#endif /* GML_CORRELATIONTENSOR_H_ */
