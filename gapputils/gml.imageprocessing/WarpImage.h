/*
 * WarpImage.h
 *
 *  Created on: Nov 7, 2014
 *      Author: tombr
 */

#ifndef GML_WARPIMAGE_H_
#define GML_WARPIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/Tensor.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imageprocessing {

struct WarpImageChecker { WarpImageChecker(); } ;

class WarpImage : public DefaultWorkflowElement<WarpImage> {

  friend class WarpImageChecker;

  InitReflectableClass(WarpImage)

  Property(Input, boost::shared_ptr<image_t>)
  Property(Deformation, boost::shared_ptr<host_tensor_t>)
  Property(VoxelSize, std::vector<int>)
  Property(Output, boost::shared_ptr<image_t>)

public:
  WarpImage();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_WARPIMAGE_H_ */
