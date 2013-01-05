/*
 * ModelWriter.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <capputils/Serializer.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;

namespace gml {
namespace convrbm4d {

BeginPropertyDefinitions(ModelWriter)
  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>())
  WorkflowProperty(Filename, Filename(), NotEmpty<Type>())
  WorkflowProperty(OutputName, Output("File"))
EndPropertyDefinitions

ModelWriter::ModelWriter() {
  setLabel("Writer");
}

void ModelWriter::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  bio::filtering_ostream file;
  file.push(boost::iostreams::gzip_compressor());
  file.push(bio::file_descriptor_sink(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for writing. Aborting!";
    return;
  }

  Serializer::WriteToFile(*getModel(), file);
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace convrbm4d */
} /* namespace gml */
