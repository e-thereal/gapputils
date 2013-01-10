/*
 * ReorientSlices.h
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#ifndef GML_REORIENTSLICES_H_
#define GML_REORIENTSLICES_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace core {

CapputilsEnumerator(SliceOrientation, Axial, Sagital, Coronal);

class ReorientSlices : public DefaultWorkflowElement<ReorientSlices> {

  InitReflectableClass(ReorientSlices)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Orientation, SliceOrientation)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  ReorientSlices();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_REORIENTSLICES_H_ */
