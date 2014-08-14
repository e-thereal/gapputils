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

  typedef std::vector<boost::shared_ptr<image_t> > v_image_t;
  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(FlattenImage)

  Property(Image, boost::shared_ptr<image_t>)
  Property(Images,boost::shared_ptr<v_image_t>)
  Property(Data, boost::shared_ptr<data_t>)
  Property(Datas, boost::shared_ptr<v_data_t>)
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
