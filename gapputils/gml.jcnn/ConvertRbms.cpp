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

namespace jcnn {

BeginPropertyDefinitions(ConvertRbms)

  ReflectableBase(DefaultWorkflowElement<ConvertRbms>)

  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(LeftCrbms, Input("LCRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(RightCrbms, Input("RCRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(LeftRbms, Input("LRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(RightRbms, Input("RRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(JointRbms, Input("JRBMs"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(InitialWeights)
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(Model, Output("JCNN"))

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

  v_data_t& labels = *getLabels();

  boost::shared_ptr<model_t> model(new model_t());

  {
    v_crbm_t& crbms = *getLeftCrbms();
    for (size_t iLayer = 0; iLayer < crbms.size(); ++iLayer) {

      model_t::cnn_layer_t layer;
      switch (crbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::ReLU1:
        case unit_type::ReLU2:
        case unit_type::ReLU4:
        case unit_type::ReLU8:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      layer.set_filters(crbms[iLayer]->filters());
      layer.set_bias(crbms[iLayer]->hidden_bias());
      layer.set_kernel_size(crbms[iLayer]->kernel_size());
      layer.set_stride_size(crbms[iLayer]->stride_size());
      layer.set_convolution_type(crbms[iLayer]->convolution_type());
      layer.set_mean(crbms[iLayer]->mean());
      layer.set_stddev(crbms[iLayer]->stddev());
      layer.set_shared_bias(crbms[iLayer]->shared_bias());

      model->append_left_cnn_layer(layer);
    }
  }

  {
    v_crbm_t& crbms = *getRightCrbms();
    for (size_t iLayer = 0; iLayer < crbms.size(); ++iLayer) {

      model_t::cnn_layer_t layer;
      switch (crbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::ReLU1:
        case unit_type::ReLU2:
        case unit_type::ReLU4:
        case unit_type::ReLU8:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << crbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      layer.set_filters(crbms[iLayer]->filters());
      layer.set_bias(crbms[iLayer]->hidden_bias());
      layer.set_kernel_size(crbms[iLayer]->kernel_size());
      layer.set_stride_size(crbms[iLayer]->stride_size());
      layer.set_convolution_type(crbms[iLayer]->convolution_type());
      layer.set_mean(crbms[iLayer]->mean());
      layer.set_stddev(crbms[iLayer]->stddev());
      layer.set_shared_bias(crbms[iLayer]->shared_bias());

      model->append_right_cnn_layer(layer);
    }
  }

  {
    v_rbm_t& rbms = *getLeftRbms();
    for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {

      if (iLayer == 0 && rbms[iLayer]->visibles_count() != getLeftCrbms()->at(getLeftCrbms()->size() - 1)->hiddens_count()) {
        dlog(Severity::Warning) << "Number of hidden units of last CRBM doesn't match the number of visible units of the first RBM. Aborting!";
        return;
      }

      model_t::nn_layer_t layer;
      switch (rbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::ReLU1:
        case unit_type::ReLU2:
        case unit_type::ReLU4:
        case unit_type::ReLU8:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << rbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      layer.set_weights(rbms[iLayer]->weights());
      layer.set_bias(rbms[iLayer]->hidden_bias());
      layer.set_mean(rbms[iLayer]->mean());
      layer.set_stddev(rbms[iLayer]->stddev());

      model->append_left_nn_layer(layer);
    }
  }

  {
    v_rbm_t& rbms = *getRightRbms();
    for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {

      if (iLayer == 0 && rbms[iLayer]->visibles_count() != getRightCrbms()->at(getRightCrbms()->size() - 1)->hiddens_count()) {
        dlog(Severity::Warning) << "Number of hidden units of last CRBM doesn't match the number of visible units of the first RBM. Aborting!";
        return;
      }

      model_t::nn_layer_t layer;
      switch (rbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::ReLU1:
        case unit_type::ReLU2:
        case unit_type::ReLU4:
        case unit_type::ReLU8:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << rbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      layer.set_weights(rbms[iLayer]->weights());
      layer.set_bias(rbms[iLayer]->hidden_bias());
      layer.set_mean(rbms[iLayer]->mean());
      layer.set_stddev(rbms[iLayer]->stddev());

      model->append_right_nn_layer(layer);
    }
  }

  {
    v_rbm_t& rbms = *getJointRbms();
    for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {

      if (iLayer == 0 && rbms[iLayer]->visibles_count() != (getLeftRbms()->at(getLeftRbms()->size() - 1)->hiddens_count() + getRightRbms()->at(getRightRbms()->size() - 1)->hiddens_count())) {
        dlog(Severity::Warning) << "Number of hidden units of last CRBM doesn't match the number of visible units of the first RBM. Aborting!";
        return;
      }

      model_t::nn_layer_t layer;
      switch (rbms[iLayer]->hiddens_type()) {
        case unit_type::Bernoulli:  layer.set_activation_function(activation_function::Sigmoid);  break;
        case unit_type::ReLU:
        case unit_type::ReLU1:
        case unit_type::ReLU2:
        case unit_type::ReLU4:
        case unit_type::ReLU8:
        case unit_type::MyReLU:     layer.set_activation_function(activation_function::ReLU); break;
        default:
          dlog(Severity::Warning) << "Unsupported hidden unit type '" << rbms[iLayer]->hiddens_type() << "'. Aborting!";
          return;
      }

      layer.set_weights(rbms[iLayer]->weights());
      layer.set_bias(rbms[iLayer]->hidden_bias());
      layer.set_mean(rbms[iLayer]->mean());
      layer.set_stddev(rbms[iLayer]->stddev());

      model->append_joint_nn_layer(layer);
    }
  }

  {
    const size_t visibleCount = getJointRbms()->at(getJointRbms()->size() - 1)->hiddens_count();
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

    model->append_joint_nn_layer(layer);
  }

  newState->setModel(model);
}

} /* namespace cnn */

} /* namespace gml */
