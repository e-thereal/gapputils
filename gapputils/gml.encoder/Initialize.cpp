/*
 * Initialize.cpp
 *
 *  Created on: Apr 14, 2015
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

#include <gapputils/attributes/GroupAttribute.h>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(Initialize)

  ReflectableBase(DefaultWorkflowElement<Initialize>)

  WorkflowProperty(FilterWidths, Group("Convolutional layers"))
  WorkflowProperty(FilterHeights, Group("Convolutional layers"))
  WorkflowProperty(FilterDepths, Group("Convolutional layers"))
  WorkflowProperty(FilterCounts, NotEmpty<Type>(), Group("Convolutional layers"))
  WorkflowProperty(WeightSparsity, Group("Optional"), Description("Indicates the percentage of non-zero weights."))
  WorkflowProperty(EncodingWeights, Group("Optional"))
  WorkflowProperty(DecodingWeights, Group("Optional"))
  WorkflowProperty(ShortcutWeights, Group("Optional"))
  WorkflowProperty(StrideWidths, Group("Convolutional layers"))
  WorkflowProperty(StrideHeights, Group("Convolutional layers"))
  WorkflowProperty(StrideDepths, Group("Convolutional layers"))
  WorkflowProperty(PoolingWidths, Group("Convolutional layers"))
  WorkflowProperty(PoolingHeights, Group("Convolutional layers"))
  WorkflowProperty(PoolingDepths, Group("Convolutional layers"))
//  WorkflowProperty(HiddenUnitCounts)
  WorkflowProperty(ConvolutionType, Enumerator<Type>(), Group("Convolutional layers"))
  WorkflowProperty(PoolingMethod, Enumerator<Type>(), Group("Convolutional layers"))
  WorkflowProperty(HiddenActivationFunction, Enumerator<Type>(), Group("Convolutional layers"))
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(NormalizeInputs, Flag())
  WorkflowProperty(Shortcuts, Enumerator<Type>())

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>(), Group("Input/output"))
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>(), Group("Input/output"))
  WorkflowProperty(Mask, Input("M"), Group("Input/output"))
  WorkflowProperty(Model, Output("ENN"), Group("Input/output"))

EndPropertyDefinitions

Initialize::Initialize() : _WeightSparsity(1), _NormalizeInputs(true), _Shortcuts(ShortcutType::NoShortcut) {
  setLabel("Init");
}

void Initialize::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  v_host_tensor_t& tensors = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

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

  /*** Add convolutional encoding layers ***/

  for (size_t iLayer = 0; iLayer < clayerCount; ++iLayer) {
    cnn_layer_t clayer;

    host_tensor_t::dim_t size = (iLayer == 0 ? tensors[0]->size() : model->cnn_encoders()[iLayer - 1]->outputs_size());

    clayer.set_visibles_size(size);
    clayer.set_activation_function(getHiddenActivationFunction());
    clayer.set_convolution_type(getConvolutionType());

    host_tensor_t::dim_t strideSize;
    strideSize[0] = getStrideWidths()[iLayer];
    strideSize[1] = getStrideHeights()[iLayer];
    strideSize[2] = getStrideDepths()[iLayer];
    strideSize[3] = 1;
    clayer.set_stride_size(strideSize);

    host_tensor_t::dim_t poolingSize;
    poolingSize[0] = getPoolingWidths()[iLayer];
    poolingSize[1] = getPoolingHeights()[iLayer];
    poolingSize[2] = getPoolingDepths()[iLayer];
    poolingSize[3] = 1;
    clayer.set_pooling_size(poolingSize);
    if (poolingSize.count() > 1)
      clayer.set_pooling_method(getPoolingMethod());
    else
      clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);

    host_tensor_t::dim_t kernelSize;
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
    v_host_tensor_t bias;
    v_host_tensor_t filters;

    random_tensor2<value_t, dimCount, false, normal<value_t> > randn(kernelSize);
    random_tensor2<value_t, dimCount, false, uniform<value_t> > randu(kernelSize);
    host_tensor_t sample;

    host_tensor_t::dim_t hiddenSize = (size + strideSize - 1) / strideSize;
    hiddenSize[dimCount - 1] = 1;

    value_t stddev = 0.5 * sqrt(2.0 / (value_t)kernelSize.count()) + 0.5 * sqrt(2.0 / (value_t)(kernelSize[0] * kernelSize[1] * kernelSize[2] * getFilterCounts()[iLayer]));
    stddev /= sqrt(_WeightSparsity);
    if (_EncodingWeights.size() > iLayer)
      stddev = _EncodingWeights[iLayer];

    for (int i = 0; i < getFilterCounts()[iLayer]; ++i) {
      sample = (stddev * randn()) * (randu() < _WeightSparsity);
      filters.push_back(boost::make_shared<host_tensor_t>(sample));
      bias.push_back(boost::make_shared<host_tensor_t>(zeros<value_t>(hiddenSize)));
    }

    clayer.set_filters(filters);
    clayer.set_bias(bias);

    model->append_cnn_encoder(clayer);

    dlog(Severity::Message) << "Added convolutional layer: visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size() << ", pooled size = " << clayer.pooled_size();
  }

  if (getShortcuts() == ShortcutType::BottomUp) {
    for (size_t iLayer = 0; iLayer < clayerCount - 1; ++iLayer) {
      cnn_layer_t clayer;

      host_tensor_t::dim_t size = (iLayer == 0 ? tensors[0]->size() : model->cnn_encoders()[iLayer - 1]->outputs_size());

      clayer.set_visibles_size(size);
      clayer.set_activation_function(getHiddenActivationFunction());
      clayer.set_convolution_type(getConvolutionType());

      host_tensor_t::dim_t strideSize;
      strideSize[0] = getStrideWidths()[iLayer];
      strideSize[1] = getStrideHeights()[iLayer];
      strideSize[2] = getStrideDepths()[iLayer];
      strideSize[3] = 1;
      clayer.set_stride_size(strideSize);

//      host_tensor_t::dim_t poolingSize;
//      poolingSize[0] = getPoolingWidths()[iLayer];
//      poolingSize[1] = getPoolingHeights()[iLayer];
//      poolingSize[2] = getPoolingDepths()[iLayer];
//      poolingSize[3] = 1;
//      clayer.set_pooling_size(poolingSize);
//      if (poolingSize.count() > 1)
//        clayer.set_pooling_method(getPoolingMethod());
//      else
//        clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
      clayer.set_pooling_size(seq<dimCount>(1));
      clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);

      host_tensor_t::dim_t kernelSize;
      kernelSize[0] = getFilterWidths()[iLayer];
      kernelSize[1] = getFilterHeights()[iLayer];
      kernelSize[2] = getFilterDepths()[iLayer];
      kernelSize[3] = size[3];
      clayer.set_kernel_size(kernelSize);
      clayer.set_mean(0.0);
      clayer.set_stddev(1.0);

      // Initialize filters and bias terms
      v_host_tensor_t bias;
      v_host_tensor_t filters;

      random_tensor2<value_t, dimCount, false, normal<value_t> > randn(kernelSize);
      random_tensor2<value_t, dimCount, false, uniform<value_t> > randu(kernelSize);
      host_tensor_t sample;

      host_tensor_t::dim_t hiddenSize = (size + strideSize - 1) / strideSize;
      hiddenSize[dimCount - 1] = 1;

      value_t stddev = 0.5 * sqrt(2.0 / (value_t)kernelSize.count()) + 0.5 * sqrt(1.0 / (value_t)(kernelSize[0] * kernelSize[1] * kernelSize[2] * getFilterCounts()[iLayer]));
      stddev /= sqrt(_WeightSparsity);
      if (_ShortcutWeights.size() > iLayer)
        stddev = _ShortcutWeights[iLayer];

      for (int i = 0; i < getFilterCounts()[iLayer]; ++i) {
        sample = (stddev * randn()) * (randu() < _WeightSparsity);
        filters.push_back(boost::make_shared<host_tensor_t>(sample));
        bias.push_back(boost::make_shared<host_tensor_t>(zeros<value_t>(hiddenSize)));
      }

      clayer.set_filters(filters);
      clayer.set_bias(bias);

      model->append_cnn_shortcut(clayer);

      dlog(Severity::Message) << "Added convolutional shortcut layer: visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size() << ", pooled size = " << clayer.pooled_size();
    }
  }

  /*** Add deconvolutional decoding layers ***/

  for (int iLayer = clayerCount - 1; iLayer >= 0; --iLayer) {
    dnn_layer_t clayer;

    host_tensor_t::dim_t size = (iLayer == 0 ? labels[0]->size() : model->cnn_encoders()[iLayer - 1]->outputs_size());
    host_tensor_t::dim_t maskSize = size;
    maskSize[dimCount - 1] = 1;

    host_tensor_t mask;
    if (iLayer == 0 && getMask()) {
      if (getMask()->size() == maskSize) {
        mask = *getMask();
      } else {
        dlog(Severity::Warning) << "Mask size does not match input size. Default mask will be used.";
      }
    }

    // Fall back
    if (mask.count() == 0)
      mask = ones<value_t>(maskSize);

    clayer.set_mask(mask);

    if (iLayer == 0)
      clayer.set_activation_function(getOutputActivationFunction());
    else
      clayer.set_activation_function(getHiddenActivationFunction());
    clayer.set_convolution_type(getConvolutionType());

    host_tensor_t::dim_t strideSize;
    strideSize[0] = getStrideWidths()[iLayer];
    strideSize[1] = getStrideHeights()[iLayer];
    strideSize[2] = getStrideDepths()[iLayer];
    strideSize[3] = 1;
    clayer.set_stride_size(strideSize);

    if (getShortcuts() == ShortcutType::BottomUp) {
      host_tensor_t::dim_t poolingSize = seq<dimCount>(1);
      if (iLayer > 0) {
        poolingSize[0] = getPoolingWidths()[iLayer - 1];
        poolingSize[1] = getPoolingHeights()[iLayer - 1];
        poolingSize[2] = getPoolingDepths()[iLayer - 1];
      }
      clayer.set_pooling_size(poolingSize);
      if (poolingSize.count() > 1) {
        clayer.set_pooling_method(getPoolingMethod());
        clayer.set_visible_pooling(true);
      } else {
        clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
      }
    } else {
      host_tensor_t::dim_t poolingSize;
      poolingSize[0] = getPoolingWidths()[iLayer];
      poolingSize[1] = getPoolingHeights()[iLayer];
      poolingSize[2] = getPoolingDepths()[iLayer];
      poolingSize[3] = 1;
      clayer.set_pooling_size(poolingSize);
      if (poolingSize.count() > 1)
        clayer.set_pooling_method(getPoolingMethod());
      else
        clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
    }

    host_tensor_t::dim_t kernelSize;
    kernelSize[0] = getFilterWidths()[iLayer];
    kernelSize[1] = getFilterHeights()[iLayer];
    kernelSize[2] = getFilterDepths()[iLayer];
    kernelSize[3] = size[3];
    clayer.set_kernel_size(kernelSize);

    clayer.set_mean(0.0);
    clayer.set_stddev(1.0);

    // Initialize filters and bias terms
    host_tensor_t bias;
    v_host_tensor_t filters;

    random_tensor2<value_t, dimCount, false, normal<value_t> > randn(kernelSize);
    random_tensor2<value_t, dimCount, false, uniform<value_t> > randu(kernelSize);
    host_tensor_t sample;

    host_tensor_t::dim_t hiddenSize = size;
    hiddenSize[dimCount - 1] = 1;

    value_t stddev;
    if ((getShortcuts() == ShortcutType::TopDown && iLayer < (int)clayerCount - 1) || (getShortcuts() == ShortcutType::BottomUp && iLayer >= 1))
      stddev = 0.5 * sqrt(2.0 / (value_t)kernelSize.count()) + 0.5 * sqrt(1.0 / (value_t)(kernelSize[0] * kernelSize[1] * kernelSize[2] * getFilterCounts()[iLayer]));
    else
      stddev = 0.5 * sqrt(2.0 / (value_t)kernelSize.count()) + 0.5 * sqrt(2.0 / (value_t)(kernelSize[0] * kernelSize[1] * kernelSize[2] * getFilterCounts()[iLayer]));
    stddev /= sqrt(_WeightSparsity);
    if ((int)_DecodingWeights.size() > iLayer)
      stddev = _DecodingWeights[iLayer];

    for (int i = 0; i < getFilterCounts()[iLayer]; ++i) {
      sample = (stddev * randn()) * (randu() < _WeightSparsity);
      filters.push_back(boost::make_shared<host_tensor_t>(sample));
    }
    clayer.set_filters(filters);

    bias = zeros<value_t>(size);
    clayer.set_bias(bias);

    model->append_dnn_decoder(clayer);

    dlog(Severity::Message) << "Added deconvolutional layer: visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size() << ", pooled size = " << clayer.pooled_size();
  }

  if (getShortcuts() == ShortcutType::TopDown) {
    for (int iLayer = clayerCount - 2; iLayer >= 0; --iLayer) {
      dnn_layer_t clayer;

      host_tensor_t::dim_t size = (iLayer == 0 ? labels[0]->size() : model->cnn_encoders()[iLayer - 1]->outputs_size());
      host_tensor_t::dim_t maskSize = size;
      maskSize[dimCount - 1] = 1;

      host_tensor_t mask;
      if (iLayer == 0 && getMask()) {
        if (getMask()->size() == maskSize) {
          mask = *getMask();
        } else {
          dlog(Severity::Warning) << "Mask size does not match input size. Default mask will be used.";
        }
      }

      // Fall back
      if (mask.count() == 0)
        mask = ones<value_t>(maskSize);

      clayer.set_mask(mask);

      if (iLayer == 0)
        clayer.set_activation_function(getOutputActivationFunction());
      else
        clayer.set_activation_function(getHiddenActivationFunction());
      clayer.set_convolution_type(getConvolutionType());

      host_tensor_t::dim_t strideSize;
      strideSize[0] = getStrideWidths()[iLayer];
      strideSize[1] = getStrideHeights()[iLayer];
      strideSize[2] = getStrideDepths()[iLayer];
      strideSize[3] = 1;
      clayer.set_stride_size(strideSize);

//      host_tensor_t::dim_t poolingSize;
//      poolingSize[0] = getPoolingWidths()[iLayer];
//      poolingSize[1] = getPoolingHeights()[iLayer];
//      poolingSize[2] = getPoolingDepths()[iLayer];
//      poolingSize[3] = 1;
//      clayer.set_pooling_size(poolingSize);
//      if (poolingSize.count() > 1)
//        clayer.set_pooling_method(getPoolingMethod());
//      else
//        clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
      clayer.set_pooling_size(seq<dimCount>(1));
      clayer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);

      host_tensor_t::dim_t kernelSize;
      kernelSize[0] = getFilterWidths()[iLayer];
      kernelSize[1] = getFilterHeights()[iLayer];
      kernelSize[2] = getFilterDepths()[iLayer];
      kernelSize[3] = size[3];
      clayer.set_kernel_size(kernelSize);

      clayer.set_mean(0.0);
      clayer.set_stddev(1.0);

      // Initialize filters and bias terms
      host_tensor_t bias;
      v_host_tensor_t filters;

      random_tensor2<value_t, dimCount, false, normal<value_t> > randn(kernelSize);
      random_tensor2<value_t, dimCount, false, uniform<value_t> > randu(kernelSize);
      host_tensor_t sample;

      host_tensor_t::dim_t hiddenSize = size;
      hiddenSize[dimCount - 1] = 1;

      value_t stddev = 0.5 * sqrt(2.0 / (value_t)kernelSize.count()) + 0.5 * sqrt(1.0 / (value_t)(kernelSize[0] * kernelSize[1] * kernelSize[2] * getFilterCounts()[iLayer]));
      if ((int)_ShortcutWeights.size() > iLayer)
        stddev = _ShortcutWeights[iLayer];
      stddev /= sqrt(_WeightSparsity);

      for (int i = 0; i < getFilterCounts()[iLayer]; ++i) {
        sample = (stddev * randn()) * (randu() < _WeightSparsity);
        filters.push_back(boost::make_shared<host_tensor_t>(sample));
      }
      clayer.set_filters(filters);

      bias = zeros<value_t>(size);
      clayer.set_bias(bias);

      model->append_dnn_shortcut(clayer);

      dlog(Severity::Message) << "Added deconvolutional shortcut layer: visible size = " << clayer.visibles_size() << ", hidden size = " << clayer.hiddens_size() << ", pooled size = " << clayer.pooled_size();
    }
  }

//  // Initialize dense layers
//  const std::vector<int>& hiddens = getHiddenUnitCounts();
//
//  for (size_t iLayer = 0; iLayer < hiddens.size() + 1; ++iLayer) {
//    const size_t visibleCount = iLayer == 0 ? model->cnn_layers()[model->cnn_layers().size() - 1]->hiddens_count() : hiddens[iLayer - 1];
//    const size_t hiddenCount = iLayer == hiddens.size() ? labels[0]->size() : hiddens[iLayer];
//
//    model_t::nn_layer_t layer;
//    if (iLayer == hiddens.size())
//      layer.set_activation_function(getOutputActivationFunction());
//    else
//      layer.set_activation_function(getHiddenActivationFunction());
//
//    matrix_t W = getInitialWeights() * random_tensor<value_t, 2, false, normal<value_t> >(visibleCount, hiddenCount);
//    matrix_t b = zeros<value_t>(1, hiddenCount);
//    layer.set_weights(W);
//    layer.set_bias(b);
//
//    matrix_t means = zeros<value_t>(1, visibleCount);
//    matrix_t stddev = ones<value_t>(1, visibleCount);
//
//    layer.set_mean(means);
//    layer.set_stddev(stddev);
//
//    model->append_nn_layer(layer);
//  }

  newState->setModel(model);
}

} /* namespace cnn */

} /* namespace gml */
