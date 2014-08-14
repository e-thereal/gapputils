/*
 * OpenModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "OpenModel.h"

#include <tbblas/deeplearn/serialize_nn_layer.hpp>

namespace gml {

namespace nn {

BeginPropertyDefinitions(OpenModel)

  ReflectableBase(DefaultWorkflowElement<OpenModel>)

  WorkflowProperty(Filename, Input("File"), Filename(), NotEmpty<Type>(), FileExists())
  WorkflowProperty(Model, Output("NN"))
  WorkflowProperty(VisibleCount, NoParameter())
  WorkflowProperty(HiddenCount, NoParameter())
  WorkflowProperty(ActivationFunction, Enumerator<Type>(), NoParameter())

EndPropertyDefinitions

OpenModel::OpenModel() : _VisibleCount(0), _HiddenCount(0) {
  setLabel("Open");
}

void OpenModel::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<nn_layer_t> model(new nn_layer_t());
  tbblas::deeplearn::deserialize(getFilename(), *model);
  newState->setModel(model);
  newState->setVisibleCount(model->visibles_count());
  newState->setHiddenCount(model->hiddens_count());
  newState->setActivationFunction(model->activation_function());
}

} /* namespace nn */

} /* namespace gml */
