/*
 * MakeImage.h
 *
 *  Created on: Jun 13, 2013
 *      Author: tombr
 */

#ifndef GML_MAKEIMAGE_H_
#define GML_MAKEIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class MakeImage : public DefaultWorkflowElement<MakeImage> {

  InitReflectableClass(MakeImage)

  Property(Data, boost::shared_ptr<std::vector<double> >)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(Image, boost::shared_ptr<image_t>)

public:
  MakeImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */
} /* namespace imaging */
} /* namespace gml */
#endif /* MAKEIMAGE_H_ */
