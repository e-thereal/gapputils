/*
 * ModelWriter.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include <tbblas/deeplearn/serialize_conv_dbn.hpp>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {

namespace dbn {

BeginPropertyDefinitions(ModelWriter)
  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("DBN"), NotNull<Type>())
  WorkflowProperty(Filename, Filename("Compressed DBN (*.dbn.gz)"), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))
EndPropertyDefinitions

ModelWriter::ModelWriter() {
  setLabel("Writer");
}

void ModelWriter::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  fs::path path(getFilename());
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());

  bio::filtering_ostream file;
  file.push(boost::iostreams::gzip_compressor());
  file.push(bio::file_descriptor_sink(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for writing. Aborting!";
    return;
  }

  tbblas::deeplearn::serialize(*getModel(), file);

  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace dbn */

} /* namespace gml */
