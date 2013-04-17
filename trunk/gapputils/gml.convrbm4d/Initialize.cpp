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

BeginPropertyDefinitions(Initialize)

  ReflectableBase(DefaultWorkflowElement<Initialize>)

  WorkflowProperty(Tensors, Input("Ts"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(FilterWidth)
  WorkflowProperty(FilterHeight)
  WorkflowProperty(FilterDepth)
  WorkflowProperty(FilterCount)
  WorkflowProperty(WeightMean)
  WorkflowProperty(WeightStddev)
  WorkflowProperty(VisibleUnitType, Enumerator<Type>())
  WorkflowProperty(HiddenUnitType, Enumerator<Type>())

  WorkflowProperty(Model, Output("CRBM"))

EndPropertyDefinitions

Initialize::Initialize()
 : _FilterWidth(9), _FilterHeight(9), _FilterDepth(9), _FilterCount(24),
   _WeightMean(0.0), _WeightStddev(1e-3)
{
  setLabel("Initialize");
}

Initialize::~Initialize() { }

InitializeChecker initializeChecker;

} /* namespace convrbm4d */
} /* namespace gml */
