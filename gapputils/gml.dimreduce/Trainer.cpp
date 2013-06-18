/*
 * Trainer.cpp
 *
 *  Created on: Jun 14, 2013
 *      Author: tombr
 */

#include "Trainer.h"

#include "dimreduce.h"

namespace gml {

namespace dimreduce {

BeginPropertyDefinitions(Trainer)
  ReflectableBase(DefaultWorkflowElement<Trainer>)

  WorkflowProperty(Inputs, Input("In"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Method, Enumerator<Type>())
  WorkflowProperty(OutputDimension)
  WorkflowProperty(Neighbors, Description("Number of neighbors used to construct the nearest neighbor graph when using LLE or Isomap"))
  WorkflowProperty(Model, Output("M"))
  WorkflowProperty(Outputs, Output("Out"))

EndPropertyDefinitions

Trainer::Trainer() : _OutputDimension(1), _Neighbors(5)
{
  setLabel("Trainer");
}

void Trainer::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<Model> model(new Model());

  trainModel(*getInputs(), getMethod(), getOutputDimension(), getNeighbors(), *model);

  newState->setModel(model);
}

} /* namespace dimreduce */

} /* namespace gml */
