/*
 * SaveModel.cpp
 *
 *  Created on: Dec 02, 2014
 *      Author: tombr
 */

#include "SaveModel.h"

#include <tbblas/deeplearn/serialize_joint_cnn.hpp>

namespace gml {

namespace jcnn {

BeginPropertyDefinitions(SaveModel)

  ReflectableBase(DefaultWorkflowElement<SaveModel>)

  WorkflowProperty(Model, Input("JCNN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Joint Convolutional Neural Network (*.jcnn)"), NotEmpty<Type>())
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
