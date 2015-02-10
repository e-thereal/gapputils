/*
 * Initialize.cpp
 *
 *  Created on: Aug 13, 2014
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

namespace cnn {

BeginPropertyDefinitions(Initialize)

  ReflectableBase(DefaultWorkflowElement<Initialize>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(FilterWidths)
  WorkflowProperty(FilterHeights)
  WorkflowProperty(FilterDepths)
  WorkflowProperty(FilterCounts, NotEmpty<Type>())
  WorkflowProperty(StrideWidths)
  WorkflowProperty(StrideHeights)
  WorkflowProperty(StrideDepths)
  WorkflowProperty(PoolingWidths)
  WorkflowProperty(PoolingHeights)
  WorkflowProperty(PoolingDepths)
  WorkflowProperty(HiddenUnitCounts)
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

  v_tensor_t& tensors = *getTrainingSet();
  v_data_t& labels = *getLabels();

  const size_t sampleCount = tensors.size();

  if (sampleCount != labels.size()) {
    dlog(Severity::Warning) << "The number of samples and labels must be the same. Aborting!";
    return;
  }

  const size_t clayerCount = getFilterCounts().size();
  if (getFilterWidths().size() != clayerCount ||
      getFilterHeights().size() != clayerCount ||
      getFilterDepths().size() != clayerCount ||
      getStrideWidths().size() != clayerCount ||
      getStrideHeights().size() != clayerCount ||
      getStrideDepths().size() != clayerCount)
  {
    dlog(Severity::Warning) << "Invalid filter or stride sizes. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());
  for (size_t iLayer = 0; iLayer < getFilterCounts().size(); ++iLayer) {

    cnn_layer_t clayer;

    clayer.set_visibles_size(tensors[0]->size());
    clayer.set_activation_function(getHiddenActivationFunction());
    clayer.set_convolution_type(getConvolutionType());

    tensor_t::dim_t strideSize;
    strideSize[0] = getStrideWidths()[iLayer];
    strideSize[1] = getStrideHeights()[iLayer];
    strideSize[2] = getStrideDepths()[iLayer];
    strideSize[3] = 1;
    clayer.set_stride_size(strideSize);

    tensor_t::dim_t size = (iLayer == 0 ? tensors[0]->size() : model->cnn_layers()[iLayer - 1]->hiddens_size());

    tensor_t::dim_t kernelSize;
    kernelSize[0] = getFilterWidths()[iLayer];
    kernelSize[1] = getFilterHeights()[iLayer];
    kernelSize[2] = getFilterDepths()[iLayer];
    kernelSize[3] = size[3];
    clayer.set_kernel_size(kernelSize);

    if (iLayer == 0 && getNormalizeInputs()) {
      const value_t count = tensors[0]->count();

      // Calculate the mean and normalize the data
      value_t mean = 0;
      for (size_t i = 0; i < tensors.size(); ++i)
        mean = mean + sum(*tensors[i]) / count;
      mean /= tensors.size();

      // Calculate the stddev and normalize the data
      value_t var = 0;
      for (size_t i = 0; i < tensors.size(); ++i)
        var += dot(*tensors[i] - mean, *tensors[i] - mean) / count;

      value_t stddev = sqrt(var / tensors.size());
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

    for (int i = 0; i < getFilterCounts()[iLayer]; ++i) {
      sample = (getInitialWeights() * randn);
      filters.push_back(boost::make_shared<tensor_t>(sample));
      bias.push_back(boost::make_shared<tensor_t>(zeros<value_t>(hiddenSize)));
    }

    clayer.set_filters(filters);
    clayer.set_bias(bias);

    model->append_cnn_layer(clayer);

    dlog(Severity::Message) << "Added convolutional layer: visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size();
  }

  // Initialize dense layers
  const std::vector<int>& hiddens = getHiddenUnitCounts();

  for (size_t iLayer = 0; iLayer < hiddens.size() + 1; ++iLayer) {
    const size_t visibleCount = iLayer == 0 ? model->cnn_layers()[model->cnn_layers().size() - 1]->hiddens_count() : hiddens[iLayer - 1];
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

    model->append_nn_layer(layer);
  }

  newState->setModel(model);
}

} /* namespace cnn */

} /* namespace gml */
