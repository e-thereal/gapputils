/*
 * FilterPatch.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "FilterPatch.h"

namespace gml {

namespace convrbm4d {

BeginPropertyDefinitions(FilterPatch)
  ReflectableBase(DefaultWorkflowElement<FilterPatch>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Inputs, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(SuperPatchWidth, Description("PatchWidth + number of patches per super patch in the x direction - 1. A value of -1 indicates a SuperPatchWidth equal to the image width."))
  WorkflowProperty(SuperPatchHeight, Description("PatchHeight + number of patches per super patch in the y direction - 1. A value of -1 indicates a SuperPatchHeight equal to the image height."))
  WorkflowProperty(SuperPatchDepth, Description("PatchDepth + number of patches per super patch in the z direction - 1. A value of -1 indicates a SuperPatchDepth equal to the image depth."))
  WorkflowProperty(FilterBatchSize, Description("Number of filters that are processed in parallel."))
  WorkflowProperty(DoubleWeights, Flag())
  WorkflowProperty(OnlyFilters, Flag())
  WorkflowProperty(SampleUnits, Flag())
  WorkflowProperty(Outputs, Output("Ts"))
EndPropertyDefinitions

FilterPatch::FilterPatch() : _SuperPatchWidth(-1), _SuperPatchHeight(-1), _SuperPatchDepth(-1), _FilterBatchSize(1), _DoubleWeights(false), _OnlyFilters(false), _SampleUnits(false) {
  setLabel("Filter");
}

FilterPatchChecker filterPatchChecker;

} /* namespace convrbm4d */

} /* namespace gml */
