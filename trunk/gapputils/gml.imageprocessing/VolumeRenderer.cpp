/*
 * VolumeRenderer.cpp
 *
 *  Created on: Jan 7, 2013
 *      Author: tombr
 */

#include "VolumeRenderer.h"

namespace gml {
namespace imageprocessing {

BeginPropertyDefinitions(VolumeRenderer)

  ReflectableBase(DefaultWorkflowElement<VolumeRenderer>)

  WorkflowProperty(Volume, Input("Vol"), NotNull<Type>())
  WorkflowProperty(Distance, Description("Distance of the viewer from the center of the volume relative to the volume size. (1.0 is equivalent to the depth of the volume)"))
  WorkflowProperty(Angle, Description("[In degree] Rotates with viewpoint around the center of the volume."))
  WorkflowProperty(SampleCount, Description("Number of samples used to sample one ray."))
  WorkflowProperty(Mode, Enumerator<Type>())
  WorkflowProperty(Orientation, Enumerator<Type>())
  WorkflowProperty(Image, Output("Img"))

EndPropertyDefinitions

VolumeRenderer::VolumeRenderer() : _Distance(5.0), _Angle(0), _SampleCount(100) {
  setLabel("VolRender");
}

VolumeRendererChecker volumeRendererChecker;

} /* namespace imageprocessing */

} /* namespace gml */
