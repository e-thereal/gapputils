/*
 * VolumeMatrix.h
 *
 *  Created on: Jan 10, 2013
 *      Author: tombr
 */

#ifndef GML_VOLUMEMATRIX_H_
#define GML_VOLUMEMATRIX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace core {

class VolumeMatrix : public DefaultWorkflowElement<VolumeMatrix> {

  InitReflectableClass(VolumeMatrix)

  Property(InputVolumes, boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > >)
  Property(MaxCount, int)
  Property(MinValue, double)
  Property(MaxValue, double)
  Property(AutoScale, bool)
  Property(ColumnCount, int)
  Property(CenterImages, bool)
  Property(CroppedWidth, int)
  Property(CroppedHeight, int)
  Property(CroppedDepth, int)
  Property(VolumeMatrix, boost::shared_ptr<image_t>)

public:
  VolumeMatrix();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_VOLUMEMATRIX_H_ */
