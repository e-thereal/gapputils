/*
 * ModelWriter.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <capputils/Serializer.h>

namespace gml {
namespace convrbm {

BeginPropertyDefinitions(ModelWriter)
  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Filename, Filename(), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))
EndPropertyDefinitions

ModelWriter::ModelWriter() {
  setLabel("Writer");
}

ModelWriter::~ModelWriter() { }

void ModelWriter::update(IProgressMonitor* monitor) const {
  Serializer::WriteToFile(*getModel(), getFilename());
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace convrbm */
} /* namespace gml */
