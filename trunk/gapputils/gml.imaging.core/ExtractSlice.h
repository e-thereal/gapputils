/*
 * ExtractSlice.h
 *
 *  Created on: Feb 15, 2013
 *      Author: tombr
 */

#ifndef GML_EXTRACTSLICE_H_
#define GML_EXTRACTSLICE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class ExtractSlice : public DefaultWorkflowElement<ExtractSlice> {

  InitReflectableClass(ExtractSlice)

  Property(Volume, boost::shared_ptr<image_t>)
  Property(SliceIndex, int)
  Property(Channels, int)
  Property(Slice, boost::shared_ptr<image_t>)

public:
  ExtractSlice();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_EXTRACTSLICE_H_ */
