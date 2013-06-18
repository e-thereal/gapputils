/*
 * Encode.cpp
 *
 *  Created on: Jun 17, 2013
 *      Author: tombr
 */

#include "Encoder.h"

#include "dimreduce.h"

namespace gml {

namespace dimreduce {

BeginPropertyDefinitions(Encoder)

  ReflectableBase(DefaultWorkflowElement<Encoder>)

  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Model, Input("M"), NotNull<Type>())
  WorkflowProperty(Direction, Enumerator<Type>())
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Encoder::Encoder() {
  setLabel("Encode");
}

void Encoder::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<std::vector<boost::shared_ptr<std::vector<double> > > > outputs(
      new std::vector<boost::shared_ptr<std::vector<double> > >());

  if (getDirection() == CodingDirection::Encode) {
    encode(*getInputs(), *getModel(), *outputs);
  } else {
    decode(*getInputs(), *getModel(), *outputs);
  }

  newState->setOutputs(outputs);
}

} /* namespace dimreduce */
} /* namespace gml */
