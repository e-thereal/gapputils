/*
 * SplitImage.h
 *
 *  Created on: Jul 25, 2012
 *      Author: tombr
 */

#ifndef GML_IMAGING_CORE_SPLITIMAGE_H
#define GML_IMAGING_CORE_SPLITIMAGE_H

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class SplitImage : public DefaultWorkflowElement<SplitImage> {

  InitReflectableClass(SplitImage)

  Property(Volume, boost::shared_ptr<image_t>)
  Property(Slices, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)

public:
  SplitImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace cv */
} /* namespace gapputils */
}
#endif /* GML_IMAGING_CORE_SPLITIMAGE_H */
