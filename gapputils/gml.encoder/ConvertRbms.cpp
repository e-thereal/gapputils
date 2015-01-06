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

namespace gml {

namespace encoder {

BeginPropertyDefinitions(ConvertRbms)

  ReflectableBase(DefaultWorkflowElement<ConvertRbms>)

  WorkflowProperty(Crbms, Input("CRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(Rbms, Input("RBMs"), Merge<Type>())
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(Model, Output("CNN"))

EndPropertyDefinitions

ConvertRbms::ConvertRbms() {
  setLabel("Convert");
}

void ConvertRbms::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  Logbook& dlog = getLogbook();

  typedef float value_t;
  typedef tbblas::tensor<value_t, 2> matrix_t;
  typedef random_tensor<value_t, 2, false, normal<value_t> > randn_t;

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

    layer.set_visibles_size(crbms[iLayer]->visibles_size());
    layer.set_filters(crbms[iLayer]->filters());
    layer.set_bias(crbms[iLayer]->hidden_bias());
    layer.set_kernel_size(crbms[iLayer]->kernel_size());
    layer.set_stride_size(crbms[iLayer]->stride_size());
    layer.set_convolution_type(crbms[iLayer]->convolution_type());
    layer.set_mean(crbms[iLayer]->mean());
    layer.set_stddev(crbms[iLayer]->stddev());
    layer.set_shared_bias(crbms[iLayer]->shared_bias());

    model->append_cnn_encoder(layer);
  }

  // Add decoders
  for (int iLayer = crbms.size() - 1; iLayer >= 0; --iLayer) {

    model_t::reverse_cnn_layer_t layer;
    switch (crbms[iLayer]->visibles_type()) {
      case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
      case unit_type::ReLU:
      case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
      case unit_type::Gaussian:   layer.set_activation_function(activation_function::Linear); break;
      default:
        dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->visibles_type() << "'. Aborting!";
        return;
    }

    // Override output activation function
    if (iLayer == 0)
      layer.set_activation_function(_OutputActivationFunction);

    layer.set_filters(crbms[iLayer]->filters());
    layer.set_bias(crbms[iLayer]->visible_bias());
    layer.set_kernel_size(crbms[iLayer]->kernel_size());
    layer.set_stride_size(crbms[iLayer]->stride_size());
    layer.set_convolution_type(crbms[iLayer]->convolution_type());
    layer.set_mean(crbms[iLayer]->mean());
    layer.set_stddev(crbms[iLayer]->stddev());
    layer.set_shared_bias(crbms[iLayer]->shared_bias());

    model->append_cnn_decoder(layer);
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

} /* namespace cnn */

} /* namespace gml */
