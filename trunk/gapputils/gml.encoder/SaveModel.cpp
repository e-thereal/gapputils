/*
 * SaveModel.cpp
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#include "SaveModel.h"

#include <tbblas/deeplearn/serialize_encoder.hpp>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(SaveModel)

  ReflectableBase(DefaultWorkflowElement<SaveModel>)

  WorkflowProperty(Model, Input("ENN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Encoder Neural Network (*.enn)"), NotEmpty<Type>())
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

} /* namespace encoder */

} /* namespace gml */
