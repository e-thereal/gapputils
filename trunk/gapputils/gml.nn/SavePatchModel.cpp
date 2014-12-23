/*
 * SavePatchModel.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#include "SavePatchModel.h"

#include <tbblas/deeplearn/serialize_nn_patch.hpp>

namespace gml {

namespace nn {

BeginPropertyDefinitions(SavePatchModel)

  ReflectableBase(DefaultWorkflowElement<SavePatchModel>)

  WorkflowProperty(Model, Input("NN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Neural Network (*.nn)"), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

SavePatchModel::SavePatchModel() {
  setLabel("Save");
}

void SavePatchModel::update(IProgressMonitor* monitor) const {
  tbblas::deeplearn::serialize(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace nn */

} /* namespace gml */
