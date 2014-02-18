/*
 * MifReader.h
 *
 *  Created on: Oct 29, 2012
 *      Author: tombr
 */

#ifndef GML_IMAGING_IO_MIFTOIMAGE_H_
#define GML_IMAGING_IO_MIFTOIMAGE_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class MifReader : public DefaultWorkflowElement<MifReader> {

  InitReflectableClass(MifReader)

  Property(MifName, std::string)
  Property(Image, boost::shared_ptr<image_t>)
  Property(MaximumIntensity, int)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(VoxelWidth, double)
  Property(VoxelHeight, double)
  Property(VoxelDepth, double)

public:
  MifReader();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_IMAGING_IO_MIFTOIMAGE_H_ */
