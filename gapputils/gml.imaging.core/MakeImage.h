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

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;
  typedef std::vector<boost::shared_ptr<image_t> > v_image_t;

  InitReflectableClass(MakeImage)

  Property(Data, boost::shared_ptr<data_t>)
  Property(Datas, boost::shared_ptr<v_data_t>)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(Image, boost::shared_ptr<image_t>)
  Property(Images, boost::shared_ptr<v_image_t>)

public:
  MakeImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */
} /* namespace imaging */
} /* namespace gml */
#endif /* MAKEIMAGE_H_ */
