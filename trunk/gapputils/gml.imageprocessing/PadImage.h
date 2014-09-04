/*
 * PadImage.h
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#ifndef GML_PADIMAGE_H_
#define GML_PADIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

class PadImage : public DefaultWorkflowElement<PadImage> {

  InitReflectableClass(PadImage)

  Property(Input, boost::shared_ptr<image_t>)
  Property(PaddedSize, std::vector<int>)
  Property(Output, boost::shared_ptr<image_t>)

public:
  PadImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_PADIMAGE_H_ */
