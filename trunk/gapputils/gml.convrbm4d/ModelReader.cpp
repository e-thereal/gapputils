/*
 * ModelReader.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "ModelReader.h"

#include <capputils/Serializer.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(ModelReader)

  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename(), FileExists())
  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(FilterWidth, NoParameter())
  WorkflowProperty(FilterHeight, NoParameter())
  WorkflowProperty(FilterDepth, NoParameter())
  WorkflowProperty(ChannelCount, NoParameter())
  WorkflowProperty(FilterCount, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader()
 : _FilterWidth(0), _FilterHeight(0), _FilterDepth(0), _ChannelCount(0), _FilterCount(0)
{
  setLabel("Reader");
}

void ModelReader::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  bio::filtering_istream file;
  file.push(boost::iostreams::gzip_decompressor());
  file.push(bio::file_descriptor_source(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for reading. Aborting!";
    return;
  }

  boost::shared_ptr<Model> model(new Model());
  Serializer::ReadFromFile(*model, file);

  newState->setModel(model);
  if (model->getFilters() && model->getFilters()->size()) {
    newState->setFilterWidth(model->getFilters()->at(0)->size()[0]);
    newState->setFilterHeight(model->getFilters()->at(0)->size()[1]);
    newState->setFilterDepth(model->getFilters()->at(0)->size()[2]);
    newState->setChannelCount(model->getFilters()->at(0)->size()[3]);
    newState->setFilterCount(model->getFilters()->size());
  }
  newState->setVisibleUnitType(model->getVisibleUnitType());
  newState->setHiddenUnitType(model->getHiddenUnitType());
}

} /* namespace convrbm4d */
} /* namespace gml */
