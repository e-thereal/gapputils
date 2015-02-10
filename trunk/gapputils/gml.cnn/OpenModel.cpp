/*
 * OpenModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "OpenModel.h"

#include <tbblas/deeplearn/serialize_cnn.hpp>

namespace gml {

namespace cnn {

BeginPropertyDefinitions(OpenModel)

  ReflectableBase(DefaultWorkflowElement<OpenModel>)

  WorkflowProperty(Filename, Input("File"), Filename("Convolutional Neural Network (*.cnn)"), NotEmpty<Type>(), FileExists())
  WorkflowProperty(Model, Output("CNN"))
  WorkflowProperty(InputSize, NoParameter())
  WorkflowProperty(FilterCounts, NoParameter())
  WorkflowProperty(HiddenCounts, NoParameter())
  WorkflowProperty(OutputCount, NoParameter())
  WorkflowProperty(LayerCount, NoParameter())
  WorkflowProperty(ConvolutionType, NoParameter())
  WorkflowProperty(HiddenActivationFunction, NoParameter())
  WorkflowProperty(OutputActivationFunction, NoParameter())

EndPropertyDefinitions

OpenModel::OpenModel() : _FilterCounts(0), _OutputCount(0), _LayerCount(0) {
  setLabel("Open");
}

void OpenModel::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<model_t> model(new model_t());
  tbblas::deeplearn::deserialize(getFilename(), *model);

  size_t clayerCount = model->cnn_layers().size(), dlayerCount = model->nn_layers().size();

  newState->setModel(model);
  newState->setLayerCount(clayerCount + dlayerCount);

  if (clayerCount) {
    newState->setInputSize(model->visibles_size());
    newState->setConvolutionType(model->cnn_layers()[0]->convolution_type());
    newState->setHiddenActivationFunction(model->cnn_layers()[0]->activation_function());

    std::vector<int> filterCounts;
    for (size_t iLayer = 0; iLayer < clayerCount; ++iLayer)
      filterCounts.push_back(model->cnn_layers()[iLayer]->filter_count());
    newState->setFilterCounts(filterCounts);
  }

  if (dlayerCount) {
    newState->setOutputCount(model->nn_layers()[dlayerCount - 1]->hiddens_count());
    newState->setOutputActivationFunction(model->nn_layers()[dlayerCount - 1]->activation_function());

    std::vector<int> hiddenCounts;
    for (size_t iLayer = 0; iLayer < dlayerCount - 1; ++iLayer)
      hiddenCounts.push_back(model->nn_layers()[iLayer]->hiddens_count());
    newState->setHiddenCounts(hiddenCounts);
  }
}

} /* namespace nn */

} /* namespace gml */
