/*
 * SaveModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "SaveModel.h"

#include <tbblas/deeplearn/serialize_nn.hpp>

namespace gml {

namespace nn {

BeginPropertyDefinitions(SaveModel)

  ReflectableBase(DefaultWorkflowElement<SaveModel>)

  WorkflowProperty(Model, Input("NN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename(), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

SaveModel::SaveModel() {
  setLabel("Save");
}

void SaveModel::update(IProgressMonitor* monitor) const {
  tbblas::deeplearn::serialize(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace nn */

} /* namespace gml */
