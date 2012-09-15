/*
 * TensorReader.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_TENSORREADER_H_
#define GAPPUTILS_ML_TENSORREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <tbblas/tensor_base.hpp>

namespace gapputils {

namespace ml {

class TensorReader : public gapputils::workflow::DefaultWorkflowElement<TensorReader> {

  typedef tbblas::tensor_base<double, 3, false> tensor_t;

  InitReflectableClass(TensorReader)

  Property(Filename, std::string)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)

public:
  TensorReader();
  virtual ~TensorReader();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTILS_ML_TENSORREADER_H_ */
