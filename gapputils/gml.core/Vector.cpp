/*
 * Vector.cpp
 *
 *  Created on: Aug 4, 2011
 *      Author: tombr
 */

#include "Vector.h"

#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(Vector)

  ReflectableBase(DefaultWorkflowElement<Vector>)
  WorkflowProperty(Vector, NotEmpty<Type>())
  WorkflowProperty(Output, Output("Out"))

EndPropertyDefinitions

Vector::Vector() {
  setLabel("Vector");
}

void Vector::update(IProgressMonitor* monitor) const {
  const std::vector<double>& input = getVector();
  newState->setOutput(boost::make_shared<std::vector<double> >(input.begin(), input.end()));
}

}

}
