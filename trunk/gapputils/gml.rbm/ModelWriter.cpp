/*
 * RbmWriter.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <capputils/Serializer.h>

namespace gml {

namespace rbm {

BeginPropertyDefinitions(ModelWriter)

  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("RBM"))
  WorkflowProperty(Filename, Filename("RBM Model (*.rbm)"))
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

ModelWriter::ModelWriter() {
  setLabel("Writer");
}

void ModelWriter::update(IProgressMonitor* monitor) const {
  Serializer::writeToFile(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

}

}
