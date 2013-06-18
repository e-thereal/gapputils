/*
 * ModelReader.cpp
 *
 *  Created on: Jun 18, 2013
 *      Author: tombr
 */

#include "ModelReader.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include "dimreduce.h"

namespace bio = boost::iostreams;

namespace gml {

namespace dimreduce {

BeginPropertyDefinitions(ModelReader)

  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename("Compressed low dimensional mapping (*.map.gz)"), FileExists())
  WorkflowProperty(Model, Output("M"))
  WorkflowProperty(Method, NoParameter())
//  WorkflowProperty(ManifoldDimensions, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader() /*: _ManifoldDimensions(0)*/ {
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
  load(*model, file);

  newState->setModel(model);
  newState->setMethod(model->getMethod());
//  newState->setManifoldDimensions(model->getModel()->low_dim());
}

} /* namespace dimreduce */

} /* namespace gml */
