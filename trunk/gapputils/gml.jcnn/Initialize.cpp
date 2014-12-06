/*
 * Initialize.cpp
 *
 *  Created on: Dec 03, 2014
 *      Author: tombr
 */

#include "Initialize.h"

#include <tbblas/random.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>

#include <tbblas/row.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/dot.hpp>

namespace gml {

namespace jcnn {

BeginPropertyDefinitions(Initialize)

  ReflectableBase(DefaultWorkflowElement<Initialize>)

  WorkflowProperty(LeftTrainingSet, Input("LD"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(RightTrainingSet, Input("RD"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(LeftFilterWidths)
  WorkflowProperty(LeftFilterHeights)
  WorkflowProperty(LeftFilterDepths)
  WorkflowProperty(LeftFilterCounts, NotEmpty<Type>())
  WorkflowProperty(LeftStrideWidths)
  WorkflowProperty(LeftStrideHeights)
  WorkflowProperty(LeftStrideDepths)
  WorkflowProperty(LeftPoolingWidths)
  WorkflowProperty(LeftPoolingHeights)
  WorkflowProperty(LeftPoolingDepths)
  WorkflowProperty(LeftHiddenUnitCounts)
  WorkflowProperty(RightFilterWidths)
  WorkflowProperty(RightFilterHeights)
  WorkflowProperty(RightFilterDepths)
  WorkflowProperty(RightFilterCounts, NotEmpty<Type>())
  WorkflowProperty(RightStrideWidths)
  WorkflowProperty(RightStrideHeights)
  WorkflowProperty(RightStrideDepths)
  WorkflowProperty(RightPoolingWidths)
  WorkflowProperty(RightPoolingHeights)
  WorkflowProperty(RightPoolingDepths)
  WorkflowProperty(RightHiddenUnitCounts)
  WorkflowProperty(JointHiddenUnitCounts)
  WorkflowProperty(InitialWeights)
  WorkflowProperty(ConvolutionType, Enumerator<Type>())
  WorkflowProperty(PoolingMethod, Enumerator<Type>())
  WorkflowProperty(HiddenActivationFunction, Enumerator<Type>())
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(NormalizeInputs, Flag())
  WorkflowProperty(Model, Output("CNN"))

EndPropertyDefinitions

Initialize::Initialize() : _InitialWeights(0.001), _NormalizeInputs(true) {
  setLabel("Init");
}

void Initialize::update(IProgressMonitor* monitor) const {

  using namespace tbblas;

  // TODO: add pooling information

  Logbook& dlog = getLogbook();

  typedef tbblas::tensor<value_t, 2> matrix_t;
//  typedef random_tensor<value_t, 2, false, normal<value_t> > rand_matrix_t;
//  typedef random_tensor<value_t, dimCount, false, normal<value_t> > rand_tensor_t;

  v_tensor_t& lefts = *getLeftTrainingSet();
  v_tensor_t& rights = *getRightTrainingSet();
  v_data_t& labels = *getLabels();

  const size_t sampleCount = labels.size();

  if (sampleCount != lefts.size() || sampleCount != rights.size()) {
    dlog(Severity::Warning) << "The number of samples and labels must be the same. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());

  {
      /* Left model */

    const size_t clayerCount = getLeftFilterCounts().size();
    if (getLeftFilterWidths().size() != clayerCount ||
        getLeftFilterHeights().size() != clayerCount ||
        getLeftFilterDepths().size() != clayerCount ||
        getLeftStrideWidths().size() != clayerCount ||
        getLeftStrideHeights().size() != clayerCount ||
        getLeftStrideDepths().size() != clayerCount)
    {
      dlog(Severity::Warning) << "Invalid filter or stride sizes. Aborting!";
      return;
    }

    for (size_t iLayer = 0; iLayer < getLeftFilterCounts().size(); ++iLayer) {

      cnn_layer_t clayer;

      const int filterCount = getLeftFilterCounts()[iLayer];

      clayer.set_activation_function(getHiddenActivationFunction());
      clayer.set_convolution_type(getConvolutionType());

      tensor_t::dim_t strideSize;
      strideSize[0] = getLeftStrideWidths()[iLayer];
      strideSize[1] = getLeftStrideHeights()[iLayer];
      strideSize[2] = getLeftStrideDepths()[iLayer];
      strideSize[3] = 1;
      clayer.set_stride_size(strideSize);

      tensor_t::dim_t size = (iLayer == 0 ? lefts[0]->size() : model->left_cnn_layers()[iLayer - 1]->hiddens_size());
      size = size / strideSize;
      size[3] = size[3] * strideSize[0] * strideSize[1] * strideSize[2];

      tensor_t::dim_t kernelSize;
      kernelSize[0] = getLeftFilterWidths()[iLayer];
      kernelSize[1] = getLeftFilterHeights()[iLayer];
      kernelSize[2] = getLeftFilterDepths()[iLayer];
      kernelSize[3] = size[3];
      clayer.set_kernel_size(kernelSize);

      if (iLayer == 0 && getNormalizeInputs()) {
        const value_t count = lefts[0]->count();

        // Calculate the mean and normalize the data
        value_t mean = 0;
        for (size_t i = 0; i < lefts.size(); ++i)
          mean = mean + sum(*lefts[i]) / count;
        mean /= lefts.size();

        // Calculate the stddev and normalize the data
        value_t var = 0;
        for (size_t i = 0; i < lefts.size(); ++i)
          var += dot(*lefts[i] - mean, *lefts[i] - mean) / count;

        value_t stddev = sqrt(var / lefts.size());
        clayer.set_mean(mean);
        clayer.set_stddev(stddev);
      } else {
        clayer.set_mean(0.0);
        clayer.set_stddev(1.0);
      }

      // Initialize filters and bias terms
      v_tensor_t bias;
      v_tensor_t filters;

      random_tensor<value_t, dimCount, false, normal<value_t> > randn(kernelSize);
      tensor_t sample;

      tensor_t::dim_t hiddenSize = size;
      hiddenSize[dimCount - 1] = 1;

      for (int i = 0; i < filterCount; ++i) {
        sample = (getInitialWeights() * randn);
        filters.push_back(boost::make_shared<tensor_t>(sample));
        bias.push_back(boost::make_shared<tensor_t>(zeros<value_t>(hiddenSize)));
      }

      clayer.set_filters(filters);
      clayer.set_bias(bias);

      model->append_left_cnn_layer(clayer);

      dlog(Severity::Message) << "Added left convolutional layer: input size = " << clayer.input_size() << ", visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size();
    }

    // Initialize dense layers
    const std::vector<int>& hiddens = getLeftHiddenUnitCounts();

    for (size_t iLayer = 0; iLayer < hiddens.size(); ++iLayer) {
      const size_t visibleCount = (iLayer == 0 ? model->left_cnn_layers()[model->left_cnn_layers().size() - 1]->hiddens_count() : hiddens[iLayer - 1]);
      const size_t hiddenCount = hiddens[iLayer];

      model_t::nn_layer_t layer;
      layer.set_activation_function(getHiddenActivationFunction());

      matrix_t W = getInitialWeights() * random_tensor<value_t, 2, false, normal<value_t> >(visibleCount, hiddenCount);
      matrix_t b = zeros<value_t>(1, hiddenCount);
      layer.set_weights(W);
      layer.set_bias(b);

      matrix_t means = zeros<value_t>(1, visibleCount);
      matrix_t stddev = ones<value_t>(1, visibleCount);

      layer.set_mean(means);
      layer.set_stddev(stddev);

      model->append_left_nn_layer(layer);
    }
  }

  {
      /* Right model */

    const size_t clayerCount = getRightFilterCounts().size();
    if (getRightFilterWidths().size() != clayerCount ||
        getRightFilterHeights().size() != clayerCount ||
        getRightFilterDepths().size() != clayerCount ||
        getRightStrideWidths().size() != clayerCount ||
        getRightStrideHeights().size() != clayerCount ||
        getRightStrideDepths().size() != clayerCount)
    {
      dlog(Severity::Warning) << "Invalid filter or stride sizes. Aborting!";
      return;
    }

    for (size_t iLayer = 0; iLayer < getRightFilterCounts().size(); ++iLayer) {

      cnn_layer_t clayer;

      const int filterCount = getRightFilterCounts()[iLayer];

      clayer.set_activation_function(getHiddenActivationFunction());
      clayer.set_convolution_type(getConvolutionType());

      tensor_t::dim_t strideSize;
      strideSize[0] = getRightStrideWidths()[iLayer];
      strideSize[1] = getRightStrideHeights()[iLayer];
      strideSize[2] = getRightStrideDepths()[iLayer];
      strideSize[3] = 1;
      clayer.set_stride_size(strideSize);

      tensor_t::dim_t size = (iLayer == 0 ? rights[0]->size() : model->right_cnn_layers()[iLayer - 1]->hiddens_size());
      size = size / strideSize;
      size[3] = size[3] * strideSize[0] * strideSize[1] * strideSize[2];

      tensor_t::dim_t kernelSize;
      kernelSize[0] = getRightFilterWidths()[iLayer];
      kernelSize[1] = getRightFilterHeights()[iLayer];
      kernelSize[2] = getRightFilterDepths()[iLayer];
      kernelSize[3] = size[3];
      clayer.set_kernel_size(kernelSize);

      if (iLayer == 0 && getNormalizeInputs()) {
        const value_t count = rights[0]->count();

        // Calculate the mean and normalize the data
        value_t mean = 0;
        for (size_t i = 0; i < rights.size(); ++i)
          mean = mean + sum(*rights[i]) / count;
        mean /= rights.size();

        // Calculate the stddev and normalize the data
        value_t var = 0;
        for (size_t i = 0; i < rights.size(); ++i)
          var += dot(*rights[i] - mean, *rights[i] - mean) / count;

        value_t stddev = sqrt(var / rights.size());
        clayer.set_mean(mean);
        clayer.set_stddev(stddev);
      } else {
        clayer.set_mean(0.0);
        clayer.set_stddev(1.0);
      }

      // Initialize filters and bias terms
      v_tensor_t bias;
      v_tensor_t filters;

      random_tensor<value_t, dimCount, false, normal<value_t> > randn(kernelSize);
      tensor_t sample;

      tensor_t::dim_t hiddenSize = size;
      hiddenSize[dimCount - 1] = 1;

      for (int i = 0; i < filterCount; ++i) {
        sample = (getInitialWeights() * randn);
        filters.push_back(boost::make_shared<tensor_t>(sample));
        bias.push_back(boost::make_shared<tensor_t>(zeros<value_t>(hiddenSize)));
      }

      clayer.set_filters(filters);
      clayer.set_bias(bias);

      model->append_right_cnn_layer(clayer);

      dlog(Severity::Message) << "Added right convolutional layer: input size = " << clayer.input_size() << ", visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size();
    }

    // Initialize dense layers
    const std::vector<int>& hiddens = getRightHiddenUnitCounts();

    for (size_t iLayer = 0; iLayer < hiddens.size(); ++iLayer) {
      const size_t visibleCount = iLayer == 0 ? model->right_cnn_layers()[model->right_cnn_layers().size() - 1]->hiddens_count() : hiddens[iLayer - 1];
      const size_t hiddenCount = hiddens[iLayer];

      model_t::nn_layer_t layer;
      layer.set_activation_function(getHiddenActivationFunction());

      matrix_t W = getInitialWeights() * random_tensor<value_t, 2, false, normal<value_t> >(visibleCount, hiddenCount);
      matrix_t b = zeros<value_t>(1, hiddenCount);
      layer.set_weights(W);
      layer.set_bias(b);

      matrix_t means = zeros<value_t>(1, visibleCount);
      matrix_t stddev = ones<value_t>(1, visibleCount);

      layer.set_mean(means);
      layer.set_stddev(stddev);

      model->append_right_nn_layer(layer);
    }
  }

  {
    /* Joint model */

    // Initialize dense layers
    const std::vector<int>& hiddens = getJointHiddenUnitCounts();

    for (size_t iLayer = 0; iLayer < hiddens.size() + 1; ++iLayer) {
      const size_t visibleCount = iLayer == 0 ?
          model->left_nn_layers()[model->left_nn_layers().size() - 1]->hiddens_count() +
          model->right_nn_layers()[model->right_nn_layers().size() - 1]->hiddens_count()
          : hiddens[iLayer - 1];
      const size_t hiddenCount = iLayer == hiddens.size() ? labels[0]->size() : hiddens[iLayer];

      model_t::nn_layer_t layer;
      if (iLayer == hiddens.size())
        layer.set_activation_function(getOutputActivationFunction());
      else
        layer.set_activation_function(getHiddenActivationFunction());

      matrix_t W = getInitialWeights() * random_tensor<value_t, 2, false, normal<value_t> >(visibleCount, hiddenCount);
      matrix_t b = zeros<value_t>(1, hiddenCount);
      layer.set_weights(W);
      layer.set_bias(b);

      matrix_t means = zeros<value_t>(1, visibleCount);
      matrix_t stddev = ones<value_t>(1, visibleCount);

      layer.set_mean(means);
      layer.set_stddev(stddev);

      model->append_joint_nn_layer(layer);
    }
  }

  newState->setModel(model);
}

} /* namespace cnn */

} /* namespace gml */
