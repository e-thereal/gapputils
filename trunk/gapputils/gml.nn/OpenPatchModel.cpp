/*
 * OpenPatchModel.cpp
 *
 *  Created on: Dec 18, 2014
 *      Author: tombr
 */

#include "OpenPatchModel.h"

#include <tbblas/deeplearn/serialize_nn_patch.hpp>

namespace gml {

namespace nn {

BeginPropertyDefinitions(OpenPatchModel)

  ReflectableBase(DefaultWorkflowElement<OpenPatchModel>)

  WorkflowProperty(Filename, Input("File"), Filename("Neural Network (*.nn)"), NotEmpty<Type>(), FileExists())
  WorkflowProperty(Model, Output("NN"))
  WorkflowProperty(PatchSize, NoParameter())
  WorkflowProperty(HiddenCounts, NoParameter())
  WorkflowProperty(OutputCount, NoParameter())
  WorkflowProperty(LayerCount, NoParameter())
  WorkflowProperty(HiddenActivationFunction, NoParameter())
  WorkflowProperty(OutputActivationFunction, NoParameter())

EndPropertyDefinitions

OpenPatchModel::OpenPatchModel() : _OutputCount(0), _LayerCount(0) {
  setLabel("Open");
}

void OpenPatchModel::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  boost::shared_ptr<patch_model_t> model(new patch_model_t());
  tbblas::deeplearn::deserialize(getFilename(), *model);

  size_t layerCount = model->model().layers().size();

  if (layerCount < 1) {
    dlog(Severity::Warning) << "The model needs to have at least one layer. Aborting!";
    return;
  }

  if (model->patch_size().prod() != (int)model->model().visibles_count()) {
    dlog(Severity::Warning) << "Number of visible units doesn't match number of elements of the patch. Aborting!";
    return;
  }

  newState->setModel(model);
  newState->setLayerCount(layerCount);
  newState->setPatchSize(model->patch_size());
  newState->setHiddenActivationFunction(model->model().layers()[0]->activation_function());

  newState->setOutputCount(model->model().layers()[layerCount - 1]->hiddens_count());
  newState->setOutputActivationFunction(model->model().layers()[layerCount - 1]->activation_function());

  std::vector<int> hiddenCounts;
  for (size_t iLayer = 0; iLayer < layerCount - 1; ++iLayer)
    hiddenCounts.push_back(model->model().layers()[iLayer]->hiddens_count());
  newState->setHiddenCounts(hiddenCounts);
}

} /* namespace nn */

} /* namespace gml */
