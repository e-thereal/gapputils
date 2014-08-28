/*
 * OpenMif.h
 *
 *  Created on: Oct 29, 2012
 *      Author: tombr
 */

#ifndef GML_IMAGING_IO_OPENMIF_H_
#define GML_IMAGING_IO_OPENMIF_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class OpenMif : public DefaultWorkflowElement<OpenMif> {

  InitReflectableClass(OpenMif)

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
  OpenMif();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

}

}

}

#endif /* GML_IMAGING_IO_OPENMIF_H_ */
