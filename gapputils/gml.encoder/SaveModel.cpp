/*
 * SaveModel.cpp
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#include "SaveModel.h"

#include <tbblas/deeplearn/serialize_encoder.hpp>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>

#include <capputils/attributes/DummyAttribute.h>
#include <capputils/EventHandler.h>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {

namespace encoder {

int SaveModel::modelId;

BeginPropertyDefinitions(SaveModel)

  ReflectableBase(DefaultWorkflowElement<SaveModel>)

  WorkflowProperty(Model, Input("ENN"), NotNull<Type>(), Dummy(modelId = Id))
  WorkflowProperty(Filename, Filename("Encoder Neural Network (*.enn *.enn.gz)"), NotEmpty<Type>())
  WorkflowProperty(AutoSave, Flag())
  WorkflowProperty(OutputName, Output("File"))

EndPropertyDefinitions

SaveModel::SaveModel() : _AutoSave(false) {
  setLabel("Save");
  Changed.connect(EventHandler<SaveModel>(this, &SaveModel::changedHandler));
}

void SaveModel::changedHandler(ObservableClass* sender, int eventId) {
  if (eventId == modelId && getAutoSave()) {
    bio::stream<bio::null_sink> nullOstream((bio::null_sink()));
    if (capputils::Verifier::Valid(*this, nullOstream)) {
      this->execute(0);
      this->writeResults();
    }
  }
}

void SaveModel::update(IProgressMonitor* monitor) const {
  fs::path path(getFilename());
  fs::create_directories(path.parent_path());

  bio::filtering_ostream file;

  if (path.extension() == ".gz")
    file.push(boost::iostreams::gzip_compressor());
  file.push(bio::file_descriptor_sink(getFilename()));

  tbblas::deeplearn::serialize(*getModel(), file);
  getHostInterface()->saveDataModel(getFilename() + ".config");
  newState->setOutputName(getFilename());
}

} /* namespace encoder */

} /* namespace gml */
