/*
 * WarpImage.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: tombr
 */

#include "WarpImage.h"

namespace gml {

namespace imageprocessing {

BeginPropertyDefinitions(WarpImage)

  ReflectableBase(DefaultWorkflowElement<WarpImage>)

  WorkflowProperty(Input, Input("I"), NotNull<Type>())
  WorkflowProperty(Deformation, Input("D"), NotNull<Type>())
  WorkflowProperty(VoxelSize)
  WorkflowProperty(Output, Output("I"))

EndPropertyDefinitions

WarpImage::WarpImage() {
  setLabel("Warp");
}

WarpImageChecker warpImageChecker;

} /* namespace imageprocessing */

} /* namespace gml */
