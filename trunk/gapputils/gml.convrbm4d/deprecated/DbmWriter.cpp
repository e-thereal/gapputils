/*
 * DbmWriter.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "DbmWriter.h"

#include <capputils/DummyAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/Serializer.h>
#include <capputils/DeprecatedAttribute.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {
namespace convrbm4d {

int DbmWriter::modelId;

BeginPropertyDefinitions(DbmWriter, Deprecated("Use gml.dbm.ModelWriter instead."))
  ReflectableBase(DefaultWorkflowElement<DbmWriter>)

  WorkflowProperty(Model, Input("DBM"), NotNull<Type>(), Dummy(modelId = Id))
  WorkflowProperty(Filename, Filename("Compressed DBM (*.dbm.gz)"), NotEmpty<Type>())
  WorkflowProperty(AutoSave, Flag(), Description("If checked, the model is saved automatically when a change is detected."))
  WorkflowProperty(OutputName, Output("File"))
EndPropertyDefinitions

DbmWriter::DbmWriter() : _AutoSave(false) {
  setLabel("Writer");
  Changed.connect(EventHandler<DbmWriter>(this, &DbmWriter::changedHandler));
}

void DbmWriter::changedHandler(ObservableClass* sender, int eventId) {
  if (eventId == modelId && getAutoSave()) {
    this->execute(0);
    this->writeResults();
  }
}

void DbmWriter::update(IProgressMonitor* monitor) const {
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

  Serializer::WriteToFile(*getModel(), file);
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace convrbm4d */

} /* namespace gml */
