/*
 * ModelReader.cpp
 *
 *  Created on: Nov 23, 2012
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
namespace convrbm4d {

BeginPropertyDefinitions(ModelReader)

  ReflectableBase(DefaultWorkflowElement<ModelReader>)

  WorkflowProperty(Filename, Input("File"), Filename("Compressed CRBM (*.crbm.gz)"), FileExists())
  WorkflowProperty(Model, Output("CRBM"))
  WorkflowProperty(TensorWidth, NoParameter())
  WorkflowProperty(TensorHeight, NoParameter())
  WorkflowProperty(TensorDepth, NoParameter())
  WorkflowProperty(FilterWidth, NoParameter())
  WorkflowProperty(FilterHeight, NoParameter())
  WorkflowProperty(FilterDepth, NoParameter())
  WorkflowProperty(ChannelCount, NoParameter())
  WorkflowProperty(FilterCount, NoParameter())
  WorkflowProperty(VisibleUnitType, NoParameter())
  WorkflowProperty(HiddenUnitType, NoParameter())
  WorkflowProperty(ConvolutionType, NoParameter())
  WorkflowProperty(Mean, NoParameter())
  WorkflowProperty(Stddev, NoParameter())

EndPropertyDefinitions

ModelReader::ModelReader()
 : _TensorWidth(0), _TensorHeight(0), _TensorDepth(0), _FilterWidth(0), _FilterHeight(0), _FilterDepth(0), _ChannelCount(0), _FilterCount(0)
{
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

  boost::shared_ptr<model_t> model(new model_t());
  tbblas::deeplearn::deserialize(file, *model);
  newState->setModel(model);

  newState->setTensorWidth(model->visible_bias().size()[0]);
  newState->setTensorHeight(model->visible_bias().size()[1]);
  newState->setTensorDepth(model->visible_bias().size()[2]);
  if (model->filters().size()) {
    newState->setFilterWidth(model->kernel_size()[0]);
    newState->setFilterHeight(model->kernel_size()[1]);
    newState->setFilterDepth(model->kernel_size()[2]);
    newState->setChannelCount(model->filters()[0]->size()[3]);
  }
  newState->setFilterCount(model->filters().size());
  newState->setVisibleUnitType(model->visibles_type());
  newState->setHiddenUnitType(model->hiddens_type());
  newState->setConvolutionType(model->convolution_type());
  newState->setMean(model->mean());
  newState->setStddev(model->stddev());
}

} /* namespace convrbm4d */

} /* namespace gml */
