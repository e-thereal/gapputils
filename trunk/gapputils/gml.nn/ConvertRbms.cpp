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

namespace nn {

/*
 *   Property(TrainingSet, boost::shared_ptr<v_data_t>)
  Property(Labels, boost::shared_ptr<v_data_t>)
  Property(Rbms, boost::shared_ptr<v_rbm_t>)
  Property(OutputActivationFunction, tbblas::deeplearn::activation_function)
  Property(Model, boost::shared_ptr<model_t>)
 */

BeginPropertyDefinitions(ConvertRbms)

  ReflectableBase(DefaultWorkflowElement<ConvertRbms>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Rbms, Input("Rbms"), NotNull<Type>(), NotEmpty<Type>(), Merge<Type>())
  WorkflowProperty(InitialWeights)
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(Model, Output("NN"))

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

  v_data_t& data = *getTrainingSet();
  v_data_t& labels = *getLabels();

  const size_t sampleCount = data.size();

  if (sampleCount != labels.size()) {
    dlog(Severity::Warning) << "The number of samples and labels must be the same. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());
  v_rbm_t& rbms = *getRbms();

  for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {

    if (iLayer == 0 && rbms[iLayer]->visibles_count() != data[0]->size()) {
      dlog(Severity::Warning) << "Number of input units doesn't match the number of visibles units of the first RBM. Aborting!";
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

    // TODO: Implement template value type
    layer.set_weights(rbms[iLayer]->weights());
    layer.set_bias(rbms[iLayer]->hidden_bias());
    layer.set_mean(rbms[iLayer]->mean());
    layer.set_stddev(rbms[iLayer]->stddev());

    model->append_layer(layer);
  }

  {
    const size_t visibleCount = rbms[rbms.size() - 1]->hiddens_count();
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

    model->append_layer(layer);
  }

  newState->setModel(model);
}

} /* namespace nn */

} /* namespace gml */
