/*
 * CorrelateTensor.h
 *
 *  Created on: Dec 10, 2013
 *      Author: tombr
 */

#ifndef GML_CORRELATETENSOR_H_
#define GML_CORRELATETENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imageprocessing {

class CorrelateTensor : public DefaultWorkflowElement<CorrelateTensor> {

  typedef float value_t;
  typedef tbblas::tensor<value_t, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;

  typedef std::vector<double> data_t;

  InitReflectableClass(CorrelateTensor)

  Property(Tensors, boost::shared_ptr<v_tensor_t>)
  Property(Data, boost::shared_ptr<data_t>)
  Property(CorrelationTensor, boost::shared_ptr<tensor_t>)

public:
  CorrelateTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_CORRELATETENSOR_H_ */
