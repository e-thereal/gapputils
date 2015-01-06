/*
 * OpenModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "OpenModel.h"

#include <tbblas/deeplearn/serialize_encoder.hpp>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(OpenModel)

  ReflectableBase(DefaultWorkflowElement<OpenModel>)

  WorkflowProperty(Filename, Input("File"), Filename("Encoder Neural Network (*.enn)"), NotEmpty<Type>(), FileExists())
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
  boost::shared_ptr<model_t> model(new model_t());
  tbblas::deeplearn::deserialize(getFilename(), *model);

  size_t clayerCount = model->cnn_encoders().size(), dlayerCount = model->nn_encoders().size();

  newState->setModel(model);
  newState->setLayerCount(clayerCount + dlayerCount + model->cnn_decoders().size() + model->nn_decoders().size());

  if (clayerCount) {
    newState->setInputSize(model->inputs_size());
    newState->setOutputSize(model->outputs_size());
    newState->setConvolutionType(model->cnn_encoders()[0]->convolution_type());
    newState->setHiddenActivationFunction(model->cnn_encoders()[0]->activation_function());
    newState->setOutputActivationFunction(model->cnn_decoders()[model->cnn_decoders().size() - 1]->activation_function());

    std::vector<int> filterCounts;
    for (size_t iLayer = 0; iLayer < clayerCount; ++iLayer)
      filterCounts.push_back(model->cnn_encoders()[iLayer]->filter_count());

    for (size_t iLayer = 0; iLayer < model->cnn_decoders().size(); ++iLayer)
      filterCounts.push_back(model->cnn_decoders()[iLayer]->filter_count());

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
