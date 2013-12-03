/*
 * FlipImage.h
 *
 *  Created on: Nov 28, 2013
 *      Author: tombr
 */

#ifndef GML_FLIPIMAGE_H_
#define GML_FLIPIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imaging {

namespace core {

CapputilsEnumerator(FlipAxis, LeftRight, AnteriorPosterior, SuperiorInferior)

class FlipImage : public DefaultWorkflowElement<FlipImage> {

  InitReflectableClass(FlipImage)

  Property(Input, boost::shared_ptr<image_t>)
  Property(FlipAxis, FlipAxis)
  Property(Output, boost::shared_ptr<image_t>)

public:
  FlipImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;

};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_FLIPIMAGE_H_ */
