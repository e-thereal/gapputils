/*
 * InitializePatch.cpp
 *
 *  Created on: Oct 16, 2014
 *      Author: tombr
 */

#include "InitializePatch.h"

#include <iostream>

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(InitializePatch, Description("Initializes a patch-based convRBM."))

  ReflectableBase(DefaultWorkflowElement<InitializePatch>)

  WorkflowProperty(Tensors, Input("Ts"))
  WorkflowProperty(FilterWidth)
  WorkflowProperty(FilterHeight)
  WorkflowProperty(FilterDepth)
  WorkflowProperty(FilterCount)
  WorkflowProperty(StrideWidth)
  WorkflowProperty(StrideHeight)
  WorkflowProperty(StrideDepth)
  WorkflowProperty(PoolingMethod, Enumerator<Type>())
  WorkflowProperty(PoolingWidth)
  WorkflowProperty(PoolingHeight)
  WorkflowProperty(PoolingDepth)
  WorkflowProperty(WeightMean)
  WorkflowProperty(WeightStddev)
  WorkflowProperty(PatchWidth, Description("Width of a training patch."))
  WorkflowProperty(PatchHeight, Description("Height of a training patch."))
  WorkflowProperty(PatchDepth, Description("Depth of a training patch."))
  WorkflowProperty(PatchChannels, Description("Number of channels of a patch."))
  WorkflowProperty(VisibleUnitType, Enumerator<Type>())
  WorkflowProperty(HiddenUnitType, Enumerator<Type>())
  WorkflowProperty(ConvolutionType, Enumerator<Type>())

  WorkflowProperty(Model, Output("CRBM"))

EndPropertyDefinitions

InitializePatch::InitializePatch()
 : _FilterWidth(9), _FilterHeight(9), _FilterDepth(9), _FilterCount(24),
   _StrideWidth(1), _StrideHeight(1), _StrideDepth(1),
   _PoolingWidth(1), _PoolingHeight(1), _PoolingDepth(1),
   _WeightMean(0.0), _WeightStddev(1e-3),
   _PatchWidth(16), _PatchHeight(16), _PatchDepth(16), _PatchChannels(1)
{
  setLabel("Init");
}

InitializePatch::~InitializePatch() { }

InitializePatchChecker initializePatchChecker;

} /* namespace convrbm4d */
} /* namespace gml */
