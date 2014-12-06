/*
 * InitializePatch.cpp
 *
 *  Created on: Dec 02, 2014
 *      Author: tombr
 */

#include "InitializePatch.h"

#include <tbblas/random.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>

#include <tbblas/row.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/dot.hpp>

namespace gml {

namespace cnn {

BeginPropertyDefinitions(InitializePatch)

  ReflectableBase(DefaultWorkflowElement<InitializePatch>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(PatchWidth)
  WorkflowProperty(PatchHeight)
  WorkflowProperty(PatchDepth)
  WorkflowProperty(FilterWidths)
  WorkflowProperty(FilterHeights)
  WorkflowProperty(FilterDepths)
  WorkflowProperty(FilterCounts, NotEmpty<Type>())
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

InitializePatch::InitializePatch() : _PatchWidth(33), _PatchHeight(33), _PatchDepth(33), _InitialWeights(0.001), _NormalizeInputs(true) {
  setLabel("Init");
}

void InitializePatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef tbblas::tensor<value_t, 2> matrix_t;
//  typedef random_tensor<value_t, 2, false, normal<value_t> > rand_matrix_t;
//  typedef random_tensor<value_t, dimCount, false, normal<value_t> > rand_tensor_t;

  v_tensor_t& tensors = *getTrainingSet();
  v_tensor_t& labels = *getLabels();

  const size_t sampleCount = tensors.size();

  if (sampleCount != labels.size()) {
    dlog(Severity::Warning) << "The number of samples and labels must be the same. Aborting!";
    return;
  }

  for (size_t i = 0; i < dimCount - 1; ++i) {
    if (tensors[0]->size()[i] != labels[0]->size()[i]) {
      dlog(Severity::Warning) << "Size of inputs and labels doesn't match. Aborting!";
      return;
    }
  }

  const size_t clayerCount = getFilterCounts().size();
  if (getFilterWidths().size() != clayerCount ||
      getFilterHeights().size() != clayerCount ||
      getFilterDepths().size() != clayerCount ||
      getPoolingWidths().size() != clayerCount ||
      getPoolingHeights().size() != clayerCount ||
      getPoolingDepths().size() != clayerCount)
  {
    dlog(Severity::Warning) << "Invalid filter or pooling sizes. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());
  for (size_t iLayer = 0; iLayer < getFilterCounts().size(); ++iLayer) {

    cnn_layer_t clayer;

    const int filterCount = getFilterCounts()[iLayer];

    clayer.set_activation_function(getHiddenActivationFunction());
    clayer.set_convolution_type(getConvolutionType());
    clayer.set_pooling_method(getPoolingMethod());

    tensor_t::dim_t poolingSize;
    poolingSize[0] = getPoolingWidths()[iLayer];
    poolingSize[1] = getPoolingHeights()[iLayer];
    poolingSize[2] = getPoolingDepths()[iLayer];
    poolingSize[3] = 1;
    clayer.set_pooling_size(poolingSize);

    tensor_t::dim_t size = (iLayer == 0 ?
        seq(getPatchWidth(), getPatchHeight(), getPatchDepth(), tensors[0]->size()[dimCount - 1]) :
        model->cnn_layers()[iLayer - 1]->pooled_size());

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

    for (int i = 0; i < filterCount; ++i) {
      sample = (getInitialWeights() * randn);
      filters.push_back(boost::make_shared<tensor_t>(sample));
      bias.push_back(boost::make_shared<tensor_t>(zeros<value_t>(hiddenSize)));
    }

    clayer.set_filters(filters);
    clayer.set_bias(bias);
    clayer.set_shared_bias(true);

    model->append_cnn_layer(clayer);

    dlog(Severity::Message) << "Added convolutional layer: input size = " << clayer.input_size() << ", visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size() << ", pooled size = " << clayer.pooled_size();
  }

  // Initialize dense layers
  const std::vector<int>& hiddens = getHiddenUnitCounts();

  for (size_t iLayer = 0; iLayer < hiddens.size() + 1; ++iLayer) {
    const size_t visibleCount = iLayer == 0 ? model->cnn_layers()[model->cnn_layers().size() - 1]->pooled_count() : hiddens[iLayer - 1];
    const size_t hiddenCount = iLayer == hiddens.size() ? labels[0]->size()[dimCount - 1] : hiddens[iLayer];

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
