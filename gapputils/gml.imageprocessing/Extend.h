/*
 * Extend.h
 *
 *  Created on: Apr 26, 2013
 *      Author: tombr
 */

#ifndef GML_EXTEND_H_
#define GML_EXTEND_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class Extend : public DefaultWorkflowElement<Extend> {

  InitReflectableClass(Extend)

  Property(Input, boost::shared_ptr<image_t>)
  Property(WidthFactor, int)
  Property(HeightFactor, int)
  Property(DepthFactor, int)
  Property(Output, boost::shared_ptr<image_t>)

public:
  Extend();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_EXTEND_H_ */
