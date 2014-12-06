/*
 * OpenModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "OpenModel.h"

#include <tbblas/deeplearn/serialize_joint_cnn.hpp>

namespace gml {

namespace jcnn {

BeginPropertyDefinitions(OpenModel)

  ReflectableBase(DefaultWorkflowElement<OpenModel>)

  WorkflowProperty(Filename, Input("File"), Filename("Joint Convolutional Neural Network (*.jcnn)"), NotEmpty<Type>(), FileExists())
  WorkflowProperty(Model, Output("JCNN"))

  WorkflowProperty(LeftInputSize, NoParameter())
  WorkflowProperty(LeftFilterCounts, NoParameter())
  WorkflowProperty(LeftHiddenCounts, NoParameter())
  WorkflowProperty(LeftConvolutionType, NoParameter())

  WorkflowProperty(RightInputSize, NoParameter())
  WorkflowProperty(RightFilterCounts, NoParameter())
  WorkflowProperty(RightHiddenCounts, NoParameter())
  WorkflowProperty(RightConvolutionType, NoParameter())

  WorkflowProperty(JointHiddenCounts, NoParameter())

  WorkflowProperty(OutputCount, NoParameter())
  WorkflowProperty(HiddenActivationFunction, NoParameter())
  WorkflowProperty(OutputActivationFunction, NoParameter())

EndPropertyDefinitions

OpenModel::OpenModel() : _LeftFilterCounts(0), _RightFilterCounts(0), _OutputCount(0) {
  setLabel("Open");
}

void OpenModel::update(IProgressMonitor* monitor) const {
  boost::shared_ptr<model_t> model(new model_t());
  tbblas::deeplearn::deserialize(getFilename(), *model);

  size_t lclayerCount = model->left_cnn_layers().size(),
      rclayerCount = model->right_cnn_layers().size(),
      ldlayerCount = model->left_nn_layers().size(),
      rdlayerCount = model->right_nn_layers().size(),
      jdlayerCount = model->joint_nn_layers().size();

  newState->setModel(model);

  if (lclayerCount) {
    newState->setLeftInputSize(model->left_input_size());
    newState->setLeftConvolutionType(model->left_cnn_layers()[0]->convolution_type());
    newState->setHiddenActivationFunction(model->left_cnn_layers()[0]->activation_function());

    std::vector<int> filterCounts;
    for (size_t iLayer = 0; iLayer < lclayerCount; ++iLayer)
      filterCounts.push_back(model->left_cnn_layers()[iLayer]->filter_count());
    newState->setLeftFilterCounts(filterCounts);
  }

  if (rclayerCount) {
    newState->setRightInputSize(model->right_input_size());
    newState->setRightConvolutionType(model->right_cnn_layers()[0]->convolution_type());

    std::vector<int> filterCounts;
    for (size_t iLayer = 0; iLayer < rclayerCount; ++iLayer)
      filterCounts.push_back(model->right_cnn_layers()[iLayer]->filter_count());
    newState->setRightFilterCounts(filterCounts);
  }

  if (ldlayerCount) {
    std::vector<int> hiddenCounts;
    for (size_t iLayer = 0; iLayer < ldlayerCount; ++iLayer)
      hiddenCounts.push_back(model->left_nn_layers()[iLayer]->hiddens_count());
    newState->setLeftHiddenCounts(hiddenCounts);
  }

  if (rdlayerCount) {
    std::vector<int> hiddenCounts;
    for (size_t iLayer = 0; iLayer < rdlayerCount; ++iLayer)
      hiddenCounts.push_back(model->right_nn_layers()[iLayer]->hiddens_count());
    newState->setRightHiddenCounts(hiddenCounts);
  }

  if (jdlayerCount) {
    newState->setOutputCount(model->joint_nn_layers()[jdlayerCount - 1]->hiddens_count());
    newState->setOutputActivationFunction(model->joint_nn_layers()[jdlayerCount - 1]->activation_function());

    std::vector<int> hiddenCounts;
    for (size_t iLayer = 0; iLayer < jdlayerCount - 1; ++iLayer)
      hiddenCounts.push_back(model->joint_nn_layers()[iLayer]->hiddens_count());
    newState->setJointHiddenCounts(hiddenCounts);
  }
}

} /* namespace nn */

} /* namespace gml */
