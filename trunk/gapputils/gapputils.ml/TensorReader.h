/*
 * TensorReader.h
 *
 *  Created on: Sep 7, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ML_TENSORREADER_H_
#define GAPPUTILS_ML_TENSORREADER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <tbblas/tensor.hpp>

namespace gapputils {

namespace ml {

class TensorReader : public gapputils::workflow::DefaultWorkflowElement<TensorReader> {

  typedef tbblas::tensor<double, 3, false> tensor_t;

  InitReflectableClass(TensorReader)

  Property(Filename, std::string)
  Property(MaxCount, int)
  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(Count, int)

public:
  TensorReader();
  virtual ~TensorReader();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTILS_ML_TENSORREADER_H_ */
