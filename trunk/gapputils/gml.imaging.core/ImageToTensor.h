/*
 * ImageToTensor.h
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#ifndef GML_IMAGETOTENSOR_H_
#define GML_IMAGETOTENSOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <tbblas/tensor.hpp>

namespace gml {

namespace imaging {

namespace core {

class ImageToTensor : public DefaultWorkflowElement<ImageToTensor> {

  typedef tbblas::tensor<float, 4> tensor_t;
  typedef std::vector<boost::shared_ptr<tensor_t> > v_tensor_t;
  typedef std::vector<boost::shared_ptr<image_t> > v_image_t;

  InitReflectableClass(ImageToTensor)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Images, boost::shared_ptr<v_image_t>)
  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Tensors, boost::shared_ptr<v_tensor_t>)

public:
  ImageToTensor();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_IMAGETOTENSOR_H_ */
