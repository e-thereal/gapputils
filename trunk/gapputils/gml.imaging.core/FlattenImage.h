/*
 * FlattenImage.h
 *
 *  Created on: 2013-05-20
 *      Author: tombr
 */

#ifndef GML_FLATTENIMAGE_H_
#define GML_FLATTENIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class FlattenImage : public DefaultWorkflowElement<FlattenImage> {

  InitReflectableClass(FlattenImage)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Data, boost::shared_ptr<std::vector<double> >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)

public:
  FlattenImage();

protected:
  void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_FLATTENIMAGE_H_ */
