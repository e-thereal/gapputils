/*
 * Initialize.cpp
 *
 *  Created on: Nov 21, 2012
 *      Author: tombr
 */

#include "Initialize.h"

#include <iostream>

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(Initialize, Description("Initializes a full-image or patch-based convRBM. Note that the mask parameter will be ignored when trained patch-based."))

  ReflectableBase(DefaultWorkflowElement<Initialize>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Mask, Input("Mask"))
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
  WorkflowProperty(VisibleUnitType, Enumerator<Type>())
  WorkflowProperty(HiddenUnitType, Enumerator<Type>())
  WorkflowProperty(ConvolutionType, Enumerator<Type>())

  WorkflowProperty(Model, Output("CRBM"))

EndPropertyDefinitions

Initialize::Initialize()
 : _FilterWidth(9), _FilterHeight(9), _FilterDepth(9), _FilterCount(24),
   _StrideWidth(1), _StrideHeight(1), _StrideDepth(1),
   _PoolingWidth(1), _PoolingHeight(1), _PoolingDepth(1),
   _WeightMean(0.0), _WeightStddev(1e-3)
{
  setLabel("Initialize");
}

Initialize::~Initialize() { }

InitializeChecker initializeChecker;

} /* namespace convrbm4d */
} /* namespace gml */
