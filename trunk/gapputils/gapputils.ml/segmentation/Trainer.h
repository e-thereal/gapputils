/*
 * Trainer.h
 *
 *  Created on: Sep 28, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLS_ML_SEGMENTATION_TRAINER_H_
#define GAPPUTLS_ML_SEGMENTATION_TRAINER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>

#include <tbblas/tensor_base.hpp>

namespace gapputils {

namespace ml {

namespace segmentation {

class Trainer : public gapputils::workflow::DefaultWorkflowElement<Trainer> {

  typedef tbblas::tensor_base<double, 3, false> tensor_t;

  InitReflectableClass(Trainer)

  Property(Tensors, boost::shared_ptr<std::vector<boost::shared_ptr<tensor_t> > >)
  Property(Image, boost::shared_ptr<image_t>)
  Property(Output, boost::shared_ptr<image_t>)

public:
  Trainer();
  virtual ~Trainer();

protected:
  virtual void update(gapputils::workflow::IProgressMonitor* monitor) const;
};

} /* namespace segmentation */

} /* namespace ml */

} /* namespace gapputils */

#endif /* GAPPUTLS_ML_SEGMENTATION_TRAINER_H_ */
