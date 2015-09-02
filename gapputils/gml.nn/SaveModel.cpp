/*
 * SaveModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "SaveModel.h"

#include <tbblas/deeplearn/serialize_nn.hpp>

#include <boost/filesystem.hpp>

namespace gml {

namespace nn {

namespace fs = boost::filesystem;

BeginPropertyDefinitions(SaveModel)

  ReflectableBase(DefaultWorkflowElement<SaveModel>)

  WorkflowProperty(Model, Input("NN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Neural Network (*.nn)"), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

SaveModel::SaveModel() {
  setLabel("Save");
}

void SaveModel::update(IProgressMonitor* monitor) const {
  fs::path path(getFilename());
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());
  tbblas::deeplearn::serialize(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace nn */

} /* namespace gml */
