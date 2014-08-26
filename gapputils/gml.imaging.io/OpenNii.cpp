/*
 * OpenNii.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: tombr
 */

#include "OpenNii.h"

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(OpenNii)

  ReflectableBase(DefaultWorkflowElement<OpenNii>)

  WorkflowProperty(Filename, Input("Nii"), Filename("Nii (*.nii)"), FileExists())
  WorkflowProperty(Image, Output("Img"))
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())
  WorkflowProperty(VoxelWidth, NoParameter(), Description("Voxel width in mm."))
  WorkflowProperty(VoxelHeight, NoParameter(), Description("Voxel height in mm."))
  WorkflowProperty(VoxelDepth, NoParameter(), Description("Voxel depth in mm."))

EndPropertyDefinitions

OpenNii::OpenNii() {
  setLabel("Nii");
}

void OpenNii::update(IProgressMonitor* monitor) const {

}

} /* namespace io */

} /* namespace imaging */

} /* namespace gml */
