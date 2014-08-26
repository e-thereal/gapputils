/*
 * OpenNii.h
 *
 *  Created on: Aug 26, 2014
 *      Author: tombr
 */

#ifndef GML_OPENNII_H_
#define GML_OPENNII_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace imaging {

namespace io {

class OpenNii : public DefaultWorkflowElement<OpenNii> {

  InitReflectableClass(OpenNii)

  Property(Filename, std::string)
  Property(Image, boost::shared_ptr<image_t>)
  Property(MaximumIntensity, int)
  Property(Width, int)
  Property(Height, int)
  Property(Depth, int)
  Property(VoxelWidth, double)
  Property(VoxelHeight, double)
  Property(VoxelDepth, double)

public:
  OpenNii();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */

#endif /* GML_OPENNII_H_ */
