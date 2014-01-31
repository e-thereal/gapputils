/*
 * ImageDuplicator.h
 *
 *  Created on: Jan 29, 2014
 *      Author: tombr
 */

#ifndef GML_IMAGEDUPLICATOR_H_
#define GML_IMAGEDUPLICATOR_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class ImageDuplicator : public DefaultWorkflowElement<ImageDuplicator> {

  InitReflectableClass(ImageDuplicator)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Count, int)
  Property(Outputs, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)

public:
  ImageDuplicator();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_IMAGEDUPLICATOR_H_ */
