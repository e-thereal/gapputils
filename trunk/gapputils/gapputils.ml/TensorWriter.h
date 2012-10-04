/*
 * TensorWriter.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_TENSORWRITER_H_
#define GAPPUTILS_ML_TENSORWRITER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <tbblas/tensor.hpp>

namespace gapputils {

namespace ml {

class TensorWriter : public gapputils::workflow::DefaultWorkflowElement<TensorWriter> {

  typedef tbblas::tensor<double, 3, false> tensor_t;

  InitReflectableClass(TensorWriter)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Filename, std::string)

public:
  TensorWriter();
  virtual ~TensorWriter();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTILS_ML_TENSORWRITER_H_ */
