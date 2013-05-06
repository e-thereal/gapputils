/*
 * Shrink.h
 *
 *  Created on: Apr 29, 2013
 *      Author: tombr
 */

#ifndef GML_SHRINK_H_
#define GML_SHRINK_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(ShrinkingMethod, Average, Maximum, Minimum);

class Shrink : public DefaultWorkflowElement<Shrink> {

  InitReflectableClass(Shrink)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(WidthFactor, int)
  Property(HeightFactor, int)
  Property(DepthFactor, int)
  Property(ShrinkingMethod, ShrinkingMethod)
  Property(OutputImage, boost::shared_ptr<image_t>)

public:
  Shrink();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_SHRINK_H_ */
