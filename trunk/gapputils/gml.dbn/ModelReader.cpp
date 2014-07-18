/*
 * ModelReader.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: tombr
 */

#include "ModelReader.h"

#include <capputils/Serializer.h>
#include <tbblas/deeplearn/serialize.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;

namespace gml {

namespace dbn {

BeginPropertyDefinitions(ModelReader)
  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename("Compressed DBN (*.dbn.gz)"), FileExists())
  WorkflowProperty(Model, Output("DBN"))
  WorkflowProperty(ConvolutionalLayers, NoParameter())
  WorkflowProperty(DenseLayers, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader() : _ConvolutionalLayers(0), _DenseLayers(0) {
  setLabel("Reader");
}

void ModelReader::update(IProgressMonitor* monitor) const {
  using namespace tbblas::deeplearn;

  Logbook& dlog = getLogbook();

  bio::filtering_istream file;
  file.push(boost::iostreams::gzip_decompressor());
  file.push(bio::file_descriptor_source(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for reading. Aborting!";
    return;
  }

  boost::shared_ptr<dbn_t> model(new dbn_t());
  tbblas::deeplearn::deserialize(file, *model);
  newState->setModel(model);
  newState->setConvolutionalLayers(model->crbms().size());
  newState->setDenseLayers(model->rbms().size());
}

} /* namespace dbn */
} /* namespace gml */
