/*
 * SaveModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "SaveModel.h"

#include <tbblas/deeplearn/serialize_cnn.hpp>

namespace gml {

namespace cnn {

BeginPropertyDefinitions(SaveModel)

  ReflectableBase(DefaultWorkflowElement<SaveModel>)

  WorkflowProperty(Model, Input("CNN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Convolutional Neural Network (*.cnn)"), NotEmpty<Type>())
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

} /* namespace cnn */

} /* namespace gml */
