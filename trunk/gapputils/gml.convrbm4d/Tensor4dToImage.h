/*
 * Tensor4dToImage.h
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#ifndef GML_CONVRBM4D_TENSOR4DTOIMAGE_H_
#define GML_CONVRBM4D_TENSOR4DTOIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include "Model.h"

namespace gml {

namespace convrbm4d {

class Tensor4dToImage : public DefaultWorkflowElement<Tensor4dToImage> {

  typedef Model::tensor_t tensor_t;

  InitReflectableClass(Tensor4dToImage)

  Property(Tensor, boost::shared_ptr<tensor_t>)
  Property(Image, boost::shared_ptr<image_t>)

public:
  Tensor4dToImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* GML_CONVRBM4D_TENSOR4DTOIMAGE_H_ */
