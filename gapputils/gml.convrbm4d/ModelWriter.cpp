/*
 * ModelWriter.cpp
 *
 *  Created on: Nov 23, 2012
 *      Author: tombr
 */

#include "ModelWriter.h"

#include <capputils/attributes/DummyAttribute.h>
#include <capputils/EventHandler.h>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>

#include <tbblas/deeplearn/serialize.hpp>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {
namespace convrbm4d {

int ModelWriter::modelId;

BeginPropertyDefinitions(ModelWriter)
  ReflectableBase(DefaultWorkflowElement<ModelWriter>)

  WorkflowProperty(Model, Input("CRBM"), NotNull<Type>(), Dummy(modelId = Id))
  WorkflowProperty(Filename, Filename("Compressed CRBM (*.crbm.gz)"), NotEmpty<Type>())
  WorkflowProperty(AutoSave, Flag(), Description("If checked, the model is saved automatically when a change is detected."))
  WorkflowProperty(OutputName, Output("File"))
EndPropertyDefinitions

ModelWriter::ModelWriter() : _AutoSave(false) {
  setLabel("Writer");
  Changed.connect(EventHandler<ModelWriter>(this, &ModelWriter::changedHandler));
}

void ModelWriter::changedHandler(ObservableClass* sender, int eventId) {
  if (eventId == modelId && getAutoSave()) {
    bio::stream<bio::null_sink> nullOstream((bio::null_sink()));
    if (capputils::Verifier::Valid(*this, nullOstream)) {
      this->execute(0);
      this->writeResults();
    }
  }
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

} /* namespace convrbm4d */
} /* namespace gml */
