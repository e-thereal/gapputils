/*
 * OpenModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "OpenModel.h"

#include <tbblas/deeplearn/serialize_encoder.hpp>

#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>

namespace bio = boost::iostreams;
namespace fs = boost::filesystem;

namespace gml {

namespace encoder {

BeginPropertyDefinitions(OpenModel)

  ReflectableBase(DefaultWorkflowElement<OpenModel>)

  WorkflowProperty(Filename, Input("File"), Filename("Encoder Neural Network (*.enn *.enn.gz)"), NotEmpty<Type>(), FileExists())
  WorkflowProperty(Model, Output("ENN"))
  WorkflowProperty(InputSize, NoParameter())
  WorkflowProperty(OutputSize, NoParameter())
  WorkflowProperty(FilterCounts, NoParameter())
  WorkflowProperty(HiddenCounts, NoParameter())
  WorkflowProperty(LayerCount, NoParameter())
  WorkflowProperty(ConvolutionType, NoParameter())
  WorkflowProperty(HiddenActivationFunction, NoParameter())
  WorkflowProperty(OutputActivationFunction, NoParameter())

EndPropertyDefinitions

OpenModel::OpenModel() : _FilterCounts(0), _LayerCount(0) {
  setLabel("Open");
}

void OpenModel::update(IProgressMonitor* monitor) const {

  Logbook& dlog = getLogbook();

  bio::filtering_istream file;
  if (fs::path(getFilename()).extension() == ".gz")
    file.push(boost::iostreams::gzip_decompressor());
  file.push(bio::file_descriptor_source(getFilename()));

  if (!file) {
    dlog(Severity::Warning) << "Can't open file '" << getFilename() << "' for reading. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());
  tbblas::deeplearn::deserialize(file, *model);

  size_t clayerCount = model->cnn_encoders().size(), dlayerCount = model->nn_encoders().size();

  newState->setModel(model);
  newState->setLayerCount(clayerCount + dlayerCount + model->dnn_decoders().size() + model->nn_decoders().size());

  if (clayerCount) {
    newState->setInputSize(model->inputs_size());
    newState->setOutputSize(model->outputs_size());
    newState->setConvolutionType(model->cnn_encoders()[0]->convolution_type());
    newState->setHiddenActivationFunction(model->cnn_encoders()[0]->activation_function());
    newState->setOutputActivationFunction(model->dnn_decoders()[model->dnn_decoders().size() - 1]->activation_function());

    std::vector<int> filterCounts;
    for (size_t iLayer = 0; iLayer < clayerCount; ++iLayer)
      filterCounts.push_back(model->cnn_encoders()[iLayer]->filter_count());

    for (size_t iLayer = 0; iLayer < model->dnn_decoders().size(); ++iLayer)
      filterCounts.push_back(model->dnn_decoders()[iLayer]->filter_count());

    newState->setFilterCounts(filterCounts);
  }

  if (dlayerCount) {

    std::vector<int> hiddenCounts;
    for (size_t iLayer = 0; iLayer < dlayerCount - 1; ++iLayer)
      hiddenCounts.push_back(model->nn_encoders()[iLayer]->hiddens_count());

    for (size_t iLayer = 0; iLayer < model->nn_decoders().size() - 1; ++iLayer)
      hiddenCounts.push_back(model->nn_decoders()[iLayer]->hiddens_count());
    newState->setHiddenCounts(hiddenCounts);
  }
}

} /* namespace encoder */

} /* namespace gml */
