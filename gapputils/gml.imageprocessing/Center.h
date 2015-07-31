/*
 * Center.h
 *
 *  Created on: Jul 23, 2015
 *      Author: tombr
 */

#ifndef GML_CENTER_H_
#define GML_CENTER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

#include <tbblas/imgproc/fmatrix4.hpp>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(CenterMethod, CenterOfGravity, BestFit)

class Center : public DefaultWorkflowElement<Center> {

  InitReflectableClass(Center)

  Property(Input, boost::shared_ptr<image_t>)
  Property(Method, CenterMethod)
  Property(RoundToNearest, bool)
  Property(Transform, boost::shared_ptr<tbblas::imgproc::fmatrix4>)

public:
  Center();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_CENTER_H_ */
