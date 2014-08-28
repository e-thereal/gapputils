/*
 * OpenModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "OpenModel.h"

#include <tbblas/deeplearn/serialize_nn.hpp>

namespace gml {

namespace nn {

BeginPropertyDefinitions(OpenModel)

  ReflectableBase(DefaultWorkflowElement<OpenModel>)

  WorkflowProperty(Filename, Input("File"), Filename("Neural Network (*.nn)"), NotEmpty<Type>(), FileExists())
  WorkflowProperty(Model, Output("NN"))
  WorkflowProperty(InputCount, NoParameter())
  WorkflowProperty(HiddenCounts, NoParameter())
  WorkflowProperty(OutputCount, NoParameter())
  WorkflowProperty(LayerCount, NoParameter())
  WorkflowProperty(HiddenActivationFunction, NoParameter())
  WorkflowProperty(OutputActivationFunction, NoParameter())

EndPropertyDefinitions

OpenModel::OpenModel() : _InputCount(0), _OutputCount(0), _LayerCount(0) {
  setLabel("Open");
}

void OpenModel::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<model_t> model(new model_t());
  tbblas::deeplearn::deserialize(getFilename(), *model);

  size_t layerCount = model->layers().size();

  newState->setModel(model);
  newState->setLayerCount(layerCount);
  if (layerCount) {
    newState->setInputCount(model->layers()[0]->visibles_count());
    newState->setHiddenActivationFunction(model->layers()[0]->activation_function());

    newState->setOutputCount(model->layers()[layerCount - 1]->hiddens_count());
    newState->setOutputActivationFunction(model->layers()[layerCount - 1]->activation_function());

    std::vector<int> hiddenCounts;
    for (size_t iLayer = 0; iLayer < layerCount - 1; ++iLayer)
      hiddenCounts.push_back(model->layers()[iLayer]->hiddens_count());
    newState->setHiddenCounts(hiddenCounts);
  }
}

} /* namespace nn */

} /* namespace gml */
