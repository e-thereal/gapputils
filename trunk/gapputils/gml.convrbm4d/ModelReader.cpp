/*
 * ModelReader.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "ModelReader.h"

#include <capputils/Serializer.h>

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(ModelReader)

  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename(), FileExists())
  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(FilterWidth, NoParameter())
  WorkflowProperty(FilterHeight, NoParameter())
  WorkflowProperty(FilterDepth, NoParameter())
  WorkflowProperty(FilterCount, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader()
 : _FilterWidth(0), _FilterHeight(0), _FilterDepth(0), _FilterCount(0)
{
  setLabel("Reader");
}

ModelReader::~ModelReader() { }

void ModelReader::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<Model> model(new Model());
  Serializer::ReadFromFile(*model, getFilename());

  newState->setModel(model);
  if (model->getFilters() && model->getFilters()->size()) {
    newState->setFilterWidth(model->getFilters()->at(0)->size()[0]);
    newState->setFilterHeight(model->getFilters()->at(0)->size()[1]);
    newState->setFilterDepth(model->getFilters()->at(0)->size()[2]);
    newState->setFilterCount(model->getFilters()->size());
  }
  newState->setVisibleUnitType(model->getVisibleUnitType());
  newState->setHiddenUnitType(model->getHiddenUnitType());
}

} /* namespace convrbm */
} /* namespace gml */
