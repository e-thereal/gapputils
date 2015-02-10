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

namespace gml {

namespace cnn {

BeginPropertyDefinitions(ConvertRbms)

  ReflectableBase(DefaultWorkflowElement<ConvertRbms>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Crbms, Input("CRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(Rbms, Input("RBMs"), Merge<Type>())
  WorkflowProperty(InitialWeights)
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(Model, Output("CNN"))

EndPropertyDefinitions

ConvertRbms::ConvertRbms() : _InitialWeights(0.001) {
  setLabel("Convert");
}

void ConvertRbms::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  Logbook& dlog = getLogbook();

  typedef float value_t;
  typedef tbblas::tensor<value_t, 2> matrix_t;
  typedef random_tensor<value_t, 2, false, normal<value_t> > randn_t;

  typedef tensor_t::dim_t dim_t;

  v_tensor_t& tensors = *getTrainingSet();
  v_data_t& labels = *getLabels();

  const size_t sampleCount = tensors.size();

  if (sampleCount != labels.size()) {
    dlog(Severity::Warning) << "The number of samples and labels must be the same. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());

  size_t currentHiddenCount = 0;

  v_crbm_t& crbms = *getCrbms();
  for (size_t iLayer = 0; iLayer < crbms.size(); ++iLayer) {

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

    if (iLayer == 0 && layer.visibles_size() != tensors[0]->size()) {
      dlog(Severity::Warning) << "Input size doesn't match the input of the first layer. Aborting!";
      return;
    }
    currentHiddenCount = layer.hiddens_count();

    model->append_cnn_layer(layer);
  }

  if (getRbms()) {
    v_rbm_t& rbms = *getRbms();
    for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {

      if (iLayer == 0 && rbms[iLayer]->visibles_count() != crbms[crbms.size() - 1]->hiddens_count()) {
        dlog(Severity::Warning) << "Number of hidden units of last CRBM doesn't match the number of visible units of the first RBM. Aborting!";
        return;
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

      currentHiddenCount = layer.hiddens_count();

      model->append_nn_layer(layer);
    }
  }

  {
    const size_t visibleCount = currentHiddenCount;
    const size_t hiddenCount = labels[0]->size();

    model_t::nn_layer_t layer;

    layer.set_activation_function(getOutputActivationFunction());

    matrix_t W = getInitialWeights() * randn_t(visibleCount, hiddenCount);
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
