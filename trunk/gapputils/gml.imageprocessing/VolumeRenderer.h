/*
 * VolumeRenderer.h
 *
 *  Created on: Jan 7, 2013
 *      Author: tombr
 */

#ifndef GML_VOLUMERENDERER_H_
#define GML_VOLUMERENDERER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(VolumeRenderMode, MaximumIntensityProjection, AverageProjection);

struct VolumeRendererChecker { VolumeRendererChecker(); };

class VolumeRenderer : public DefaultWorkflowElement<VolumeRenderer> {
  friend class VolumeRendererChecker;

  InitReflectableClass(VolumeRenderer)

  Property(Volume, boost::shared_ptr<image_t>)
  Property(Distance, double)
  Property(Angle, double)
  Property(SampleCount, int)
  Property(Mode, VolumeRenderMode)
  Property(Image, boost::shared_ptr<image_t>)

public:
  VolumeRenderer();

protected:
  virtual void update(IProgressMonitor* monitor) const;
};

} /* namespace imageprocessing */

} /* namespace gml */

#endif /* GML_VOLUMERENDERER_H_ */
