/*
 * ImageToTensor4d.h
 *
 *  Created on: Jan 3, 2013
 *      Author: tombr
 */

#ifndef GML_CONVRBM4D_IMAGETOTENSOR4D_H_
#define GML_CONVRBM4D_IMAGETOTENSOR4D_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include "../Model.h"

namespace gml {

namespace convrbm4d {

class ImageToTensor4d : public DefaultWorkflowElement<ImageToTensor4d> {

  typedef model_t::host_tensor_t tensor_t;

  InitReflectableClass(ImageToTensor4d)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Tensor, boost::shared_ptr<tensor_t>)

public:
  ImageToTensor4d();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace convrbm4d */
} /* namespace gml */
#endif /* GML_CONVRBM4D_IMAGETOTENSOR4D_H_ */
