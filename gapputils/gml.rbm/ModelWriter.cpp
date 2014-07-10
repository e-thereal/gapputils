/*
 * RbmWriter.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <capputils/Serializer.h>
#include <tbblas/deeplearn/serialize.hpp>

namespace gml {

namespace rbm {

BeginPropertyDefinitions(ModelWriter)

  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("RBM"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("RBM Model (*.rbm)"), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

ModelWriter::ModelWriter() {
  setLabel("Writer");
}

void ModelWriter::update(IProgressMonitor* monitor) const {
  tbblas::deeplearn::serialize(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

}

}
