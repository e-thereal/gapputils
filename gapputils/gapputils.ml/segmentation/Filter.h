/*
 * Filter.h
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLS_ML_SEGMENTATION_FILTER_H_
#define GAPPUTLS_ML_SEGMENTATION_FILTER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

#include <tbblas/tensor.hpp>

namespace gapputils {

namespace ml {

namespace segmentation {

class Filter : public gapputils::workflow::DefaultWorkflowElement<Filter> {

  typedef tbblas::tensor<double, 3, false> tensor_t;

  InitReflectableClass(Filter)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Image, boost::shared_ptr<image_t>)
  Property(Padded, boost::shared_ptr<image_t>)
  Property(Centered, boost::shared_ptr<image_t>)
  Property(Output, boost::shared_ptr<image_t>)

public:
  Filter();
  virtual ~Filter();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace segmentation */

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTLS_ML_SEGMENTATION_FILTER_H_ */
