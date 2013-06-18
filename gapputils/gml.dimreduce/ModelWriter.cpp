/*
 * ModelWriter.cpp
 *
 *  Created on: Jun 18, 2013
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include "dimreduce.h"

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {

namespace dimreduce {

BeginPropertyDefinitions(ModelWriter)

  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("M"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Compressed low dimensional mapping (*.map.gz)"), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

ModelWriter::ModelWriter() {
  setLabel("Writer");
}

void ModelWriter::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  fs::path path(getFilename());
  fs::create_directories(path.parent_path());

  bio::filtering_ostream file;
  file.push(boost::iostreams::gzip_compressor());
  file.push(bio::file_descriptor_sink(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for writing. Aborting!";
    return;
  }

  save(*getModel(), file);
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace dimreduce */

} /* namespace gml */
