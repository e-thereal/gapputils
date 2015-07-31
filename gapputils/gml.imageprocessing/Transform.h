/*
 * Transform.h
 *
 *  Created on: Jul 23, 2015
 *      Author: tombr
 */

#ifndef GML_TRANSFORM_H_
#define GML_TRANSFORM_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <tbblas/imgproc/fmatrix4.hpp>

namespace gml {

namespace imageprocessing {

class Transform : public DefaultWorkflowElement<Transform> {

  InitReflectableClass(Transform)

  Property(Input, boost::shared_ptr<image_t>)
  Property(Transform, boost::shared_ptr<tbblas::imgproc::fmatrix4>)
  Property(Output, boost::shared_ptr<image_t>)

public:
  Transform();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */
#endif /* GML_TRANSFORM_H_ */
