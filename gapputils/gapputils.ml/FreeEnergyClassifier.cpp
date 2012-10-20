/*
 * FreeEnergyClassifier.cpp
 *
 *  Created on: Oct 19, 2012
 *      Author: tombr
 */

#include "FreeEnergyClassifier.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(FreeEnergyClassifier)

  ReflectableBase(workflow::DefaultWorkflowElement<FreeEnergyClassifier>)

  WorkflowProperty(Conditionals, Input("C"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Rbm, Input("Rbm"), NotNull<Type>())
  WorkflowProperty(FeatureCount, NotEqual<Type>(0))
  WorkflowProperty(MakeBernoulli)
  WorkflowProperty(Differences, Output("Diff"))

EndPropertyDefinitions

FreeEnergyClassifier::FreeEnergyClassifier() : _FeatureCount(0), _MakeBernoulli(false) {
  setLabel("FreeEnergy");
}

FreeEnergyClassifier::~FreeEnergyClassifier() {
}

} /* namespace ml */

} /* namespace gapputils */
