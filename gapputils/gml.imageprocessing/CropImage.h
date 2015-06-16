/*
 * CropImage.h
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#ifndef GML_CROPIMAGE_H_
#define GML_CROPIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class CropImage : public DefaultWorkflowElement<CropImage> {

  InitReflectableClass(CropImage)

  Property(Input, boost::shared_ptr<image_t>)
  Property(TopLeft, std::vector<int>)
  Property(CroppedSize, std::vector<int>)
  Property(Output, boost::shared_ptr<image_t>)

public:
  CropImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_PADIMAGE_H_ */
