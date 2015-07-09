/*
 * ConvertRbms.cpp
 *
 *  Created on: 2014-08-16
 *      Author: tombr
 */

#include "ConvertRbms.h"

#include <capputils/attributes/MergeAttribute.h>

#include <tbblas/random.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/zeros.hpp>

#include <tbblas/row.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/trans.hpp>

#include <gapputils/Tensor.h>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(ConvertRbms)

  ReflectableBase(DefaultWorkflowElement<ConvertRbms>)

  WorkflowProperty(Crbms, Input("CRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(Rbms, Input("RBMs"), Merge<Type>())
  WorkflowProperty(FirstInputChannel)
  WorkflowProperty(InputChannelCount, Description("A value of -1 indicates using the maximum number of channels."))
  WorkflowProperty(FirstOutputChannel)
  WorkflowProperty(OutputChannelCount, Description("A value of -1 indicates using the maximum number of channels."))
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(Shortcuts, Enumerator<Type>())
  WorkflowProperty(Model, Output("ENN"))

EndPropertyDefinitions

ConvertRbms::ConvertRbms()
 : _FirstInputChannel(0), _InputChannelCount(-1),
   _FirstOutputChannel(0), _OutputChannelCount(-1),
   _Shortcuts(ShortcutType::NoShortcut)
{
  setLabel("Convert");
}

void ConvertRbms::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  Logbook& dlog = getLogbook();

  typedef float value_t;
  typedef tbblas::tensor<value_t, 2> matrix_t;
  typedef random_tensor<value_t, 2, false, normal<value_t> > randn_t;

  typedef host_tensor_t::dim_t dim_t;
  const int dimCount = host_tensor_t::dimCount;

  boost::shared_ptr<model_t> model(new model_t());

  v_crbm_t& crbms = *getCrbms();

  // Add encoders
  for (size_t iLayer = 0; iLayer < crbms.size(); ++iLayer) {

    if (iLayer > 0 && crbms[iLayer]->visibles_size() != crbms[iLayer - 1]->outputs_size()) {
      dlog(Severity::Warning) << "Number of hidden units of the previous CRBM doesn't match the number of visible units of the current CRBM. Aborting!";
      return;
    }

    model_t::cnn_layer_t layer;
    switch (crbms[iLayer]->hiddens_type()) {
      case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
      case unit_type::ReLU:
      case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
      default:
        dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->hiddens_type() << "'. Aborting!";
        return;
    }

    if (iLayer == 0) {
      dim_t visibles_size = crbms[iLayer]->visibles_size(), kernel_size = crbms[iLayer]->kernel_size();

      int firstChannel = getFirstInputChannel();
      int channelCount = visibles_size[dimCount - 1] - firstChannel;
      if (getInputChannelCount() > 0)
        channelCount = std::min(channelCount, getInputChannelCount());

      kernel_size[dimCount - 1] = visibles_size[dimCount - 1] = channelCount;

      layer.set_visibles_size(visibles_size);
      layer.set_kernel_size(kernel_size);

      v_host_tensor_t& filters = crbms[iLayer]->filters();
      v_host_tensor_t inputFilters;

      dim_t pos = seq<dimCount>(0);
      pos[dimCount - 1] = firstChannel;

      for (size_t iFilter = 0; iFilter < filters.size(); ++iFilter) {
        host_tensor_t& filter = *filters[iFilter];
        inputFilters.push_back(boost::make_shared<host_tensor_t>(filter[pos, kernel_size]));
      }
      layer.set_filters(inputFilters);

    } else {
      layer.set_filters(crbms[iLayer]->filters());
      layer.set_visibles_size(crbms[iLayer]->visibles_size());
      layer.set_kernel_size(crbms[iLayer]->kernel_size());
    }

    layer.set_bias(crbms[iLayer]->hidden_bias());
    layer.set_stride_size(crbms[iLayer]->stride_size());
    layer.set_convolution_type(crbms[iLayer]->convolution_type());
    layer.set_pooling_method(crbms[iLayer]->pooling_method());
    layer.set_pooling_size(crbms[iLayer]->pooling_size());
    layer.set_mean(crbms[iLayer]->mean());
    layer.set_stddev(crbms[iLayer]->stddev());
    layer.set_shared_bias(crbms[iLayer]->shared_bias());

    model->append_cnn_encoder(layer);
  }

  if (getShortcuts() == ShortcutType::BottomUp) {
    // Add encoders
    for (size_t iLayer = 0; iLayer < crbms.size() - 1; ++iLayer) {

      if (iLayer > 0 && crbms[iLayer]->visibles_size() != crbms[iLayer - 1]->outputs_size()) {
        dlog(Severity::Warning) << "Number of hidden units of the previous CRBM doesn't match the number of visible units of the current CRBM. Aborting!";
        return;
      }

      model_t::cnn_layer_t layer;
      switch (crbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      if (iLayer == 0) {
        dim_t visibles_size = crbms[iLayer]->visibles_size(), kernel_size = crbms[iLayer]->kernel_size();

        int firstChannel = getFirstInputChannel();
        int channelCount = visibles_size[dimCount - 1] - firstChannel;
        if (getInputChannelCount() > 0)
          channelCount = std::min(channelCount, getInputChannelCount());

        kernel_size[dimCount - 1] = visibles_size[dimCount - 1] = channelCount;

        layer.set_visibles_size(visibles_size);
        layer.set_kernel_size(kernel_size);

        v_host_tensor_t& filters = crbms[iLayer]->filters();
        v_host_tensor_t inputFilters;

        dim_t pos = seq<dimCount>(0);
        pos[dimCount - 1] = firstChannel;

        for (size_t iFilter = 0; iFilter < filters.size(); ++iFilter) {
          host_tensor_t& filter = *filters[iFilter];
          inputFilters.push_back(boost::make_shared<host_tensor_t>(filter[pos, kernel_size]));
        }
        layer.set_filters(inputFilters);

      } else {
        layer.set_filters(crbms[iLayer]->filters());
        layer.set_visibles_size(crbms[iLayer]->visibles_size());
        layer.set_kernel_size(crbms[iLayer]->kernel_size());
      }

      layer.set_bias(crbms[iLayer]->hidden_bias());
      layer.set_stride_size(crbms[iLayer]->stride_size());
      layer.set_convolution_type(crbms[iLayer]->convolution_type());
//      layer.set_pooling_method(crbms[iLayer]->pooling_method());
//      layer.set_pooling_size(crbms[iLayer]->pooling_size());
      layer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
      layer.set_pooling_size(seq<dimCount>(1));
      layer.set_mean(crbms[iLayer]->mean());
      layer.set_stddev(crbms[iLayer]->stddev());
      layer.set_shared_bias(crbms[iLayer]->shared_bias());

      model->append_cnn_shortcut(layer);
    }
  }

  // Add decoders
  for (int iLayer = crbms.size() - 1; iLayer >= 0; --iLayer) {

    model_t::dnn_layer_t layer;
    switch (crbms[iLayer]->visibles_type()) {
      case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
      case unit_type::ReLU:
      case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
      case unit_type::Gaussian:   layer.set_activation_function(activation_function::Linear); break;
      default:
        dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->visibles_type() << "'. Aborting!";
        return;
    }

    if (iLayer == 0) {
      layer.set_activation_function(_OutputActivationFunction);

      if (_OutputActivationFunction == tbblas::deeplearn::activation_function::Linear) {
        layer.set_mean(crbms[iLayer]->mean());
        layer.set_stddev(crbms[iLayer]->stddev());
      } else {
        layer.set_mean(0);
        layer.set_stddev(1);
      }

      dim_t visibles_size = crbms[iLayer]->visibles_size(), kernel_size = crbms[iLayer]->kernel_size();

      int firstChannel = getFirstOutputChannel();
      int channelCount = visibles_size[dimCount - 1] - firstChannel;
      if (getOutputChannelCount() > 0)
        channelCount = std::min(channelCount, getOutputChannelCount());

      kernel_size[dimCount - 1] = visibles_size[dimCount - 1] = channelCount;

      layer.set_kernel_size(kernel_size);

      v_host_tensor_t& filters = crbms[iLayer]->filters();
      v_host_tensor_t inputFilters;

      dim_t pos = seq<dimCount>(0);
      pos[dimCount - 1] = firstChannel;

      for (size_t iFilter = 0; iFilter < filters.size(); ++iFilter) {
        host_tensor_t& filter = *filters[iFilter];
        inputFilters.push_back(boost::make_shared<host_tensor_t>(filter[pos, kernel_size]));
      }
      layer.set_filters(inputFilters);

      host_tensor_t bias = crbms[iLayer]->visible_bias();
      host_tensor_t outputBias = bias[pos, visibles_size];
      layer.set_bias(outputBias);
    } else {
      layer.set_filters(crbms[iLayer]->filters());
      layer.set_kernel_size(crbms[iLayer]->kernel_size());
      layer.set_bias(crbms[iLayer]->visible_bias());
      layer.set_mean(crbms[iLayer]->mean());
      layer.set_stddev(crbms[iLayer]->stddev());
    }

    layer.set_mask(crbms[iLayer]->mask());
    layer.set_stride_size(crbms[iLayer]->stride_size());
    layer.set_convolution_type(crbms[iLayer]->convolution_type());
    if (getShortcuts() == ShortcutType::BottomUp) {
      if (iLayer > 0) {
        layer.set_pooling_size(crbms[iLayer - 1]->pooling_size());
        if (crbms[iLayer - 1]->pooling_size().prod() > 1) {
          layer.set_pooling_method(crbms[iLayer - 1]->pooling_method());
          layer.set_visible_pooling(true);
        } else {
          layer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
        }
      } else {
        layer.set_pooling_method(tbblas::deeplearn::pooling_method::NoPooling);
        layer.set_pooling_size(seq<dimCount>(1));
      }
    } else {
      layer.set_pooling_method(crbms[iLayer]->pooling_method());
      layer.set_pooling_size(crbms[iLayer]->pooling_size());
    }
    layer.set_shared_bias(crbms[iLayer]->shared_bias());

    model->append_dnn_decoder(layer);
  }

  // Add shortcuts
  if (getShortcuts() == ShortcutType::TopDown) {
    for (int iLayer = crbms.size() - 2; iLayer >= 0; --iLayer) {

      model_t::dnn_layer_t layer;
      switch (crbms[iLayer]->visibles_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        case unit_type::Gaussian:   layer.set_activation_function(activation_function::Linear); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->visibles_type() << "'. Aborting!";
          return;
      }

      if (iLayer == 0) {
        layer.set_activation_function(_OutputActivationFunction);

        if (_OutputActivationFunction == tbblas::deeplearn::activation_function::Linear) {
          layer.set_mean(crbms[iLayer]->mean());
          layer.set_stddev(crbms[iLayer]->stddev());
        } else {
          layer.set_mean(0);
          layer.set_stddev(1);
        }

        dim_t visibles_size = crbms[iLayer]->visibles_size(), kernel_size = crbms[iLayer]->kernel_size();

        int firstChannel = getFirstOutputChannel();
        int channelCount = visibles_size[dimCount - 1] - firstChannel;
        if (getOutputChannelCount() > 0)
          channelCount = std::min(channelCount, getOutputChannelCount());

        kernel_size[dimCount - 1] = visibles_size[dimCount - 1] = channelCount;

        layer.set_kernel_size(kernel_size);

        v_host_tensor_t& filters = crbms[iLayer]->filters();
        v_host_tensor_t inputFilters;

        dim_t pos = seq<dimCount>(0);
        pos[dimCount - 1] = firstChannel;

        for (size_t iFilter = 0; iFilter < filters.size(); ++iFilter) {
          host_tensor_t& filter = *filters[iFilter];
          inputFilters.push_back(boost::make_shared<host_tensor_t>(filter[pos, kernel_size]));
        }
        layer.set_filters(inputFilters);

        host_tensor_t bias = crbms[iLayer]->visible_bias();
        host_tensor_t outputBias = bias[pos, visibles_size];
        layer.set_bias(outputBias);
      } else {
        layer.set_filters(crbms[iLayer]->filters());
        layer.set_kernel_size(crbms[iLayer]->kernel_size());
        layer.set_bias(crbms[iLayer]->visible_bias());
        layer.set_mean(crbms[iLayer]->mean());
        layer.set_stddev(crbms[iLayer]->stddev());
      }

      layer.set_mask(crbms[iLayer]->mask());
      layer.set_stride_size(crbms[iLayer]->stride_size());
      layer.set_convolution_type(crbms[iLayer]->convolution_type());
      layer.set_pooling_method(crbms[iLayer]->pooling_method());
      layer.set_pooling_size(crbms[iLayer]->pooling_size());
      layer.set_shared_bias(crbms[iLayer]->shared_bias());

      model->append_dnn_shortcut(layer);
    }
  }

  if (getRbms()) {
    v_rbm_t& rbms = *getRbms();
    for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {

      if (iLayer == 0) {
        if (rbms[iLayer]->visibles_count() != crbms[crbms.size() - 1]->hiddens_count()) {
          dlog(Severity::Warning) << "Number of hidden units of last CRBM doesn't match the number of visible units of the first RBM. Aborting!";
          return;
        }
      } else {
        if (rbms[iLayer]->visibles_count() != rbms[iLayer - 1]->hiddens_count()) {
          dlog(Severity::Warning) << "Number of hidden units of the previous RBM doesn't match the number of visible units of the current RBM. Aborting!";
          return;
        }
      }

      model_t::nn_layer_t layer;
      switch (rbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << rbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      layer.set_weights(rbms[iLayer]->weights());
      layer.set_bias(rbms[iLayer]->hidden_bias());
      layer.set_mean(rbms[iLayer]->mean());
      layer.set_stddev(rbms[iLayer]->stddev());

      model->append_nn_encoder(layer);
    }

    for (int iLayer = rbms.size() - 1; iLayer >= 0; --iLayer) {

      model_t::nn_layer_t layer;
      switch (rbms[iLayer]->visibles_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << rbms[iLayer]->visibles_type() << "'. Aborting!";
          return;
      }

      matrix_t temp = rbms[iLayer]->weights();
      matrix_t weights = trans(temp);
      layer.set_weights(weights);
      layer.set_bias(rbms[iLayer]->visible_bias());
      layer.set_mean(rbms[iLayer]->mean());
      layer.set_stddev(rbms[iLayer]->stddev());

      model->append_nn_decoder(layer);
    }
  }

  newState->setModel(model);
}

} /* namespace encoder */

} /* namespace gml */
