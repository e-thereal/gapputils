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

namespace nn {

BeginPropertyDefinitions(Initialize)

  ReflectableBase(DefaultWorkflowElement<Initialize>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(HiddenUnitCounts)
  WorkflowProperty(InitialWeights)
  WorkflowProperty(HiddenActivationFunction, Enumerator<Type>())
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(NormalizeInputs, Flag())
  WorkflowProperty(Model, Output("NN"))

EndPropertyDefinitions

Initialize::Initialize() : _InitialWeights(0.001), _NormalizeInputs(true) {
  setLabel("Init");
}

void Initialize::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

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
  const std::vector<int>& hiddens = getHiddenUnitCounts();

  for (size_t iLayer = 0; iLayer < hiddens.size() + 1; ++iLayer) {
    const size_t visibleCount = iLayer == 0 ? data[0]->size() : hiddens[iLayer - 1];
    const size_t hiddenCount = iLayer == hiddens.size() ? labels[0]->size() : hiddens[iLayer];

    model_t::nn_layer_t layer;
    if (iLayer == hiddens.size())
      layer.set_activation_function(getOutputActivationFunction());
    else
      layer.set_activation_function(getHiddenActivationFunction());

    matrix_t W = getInitialWeights() * randn_t(visibleCount, hiddenCount);
    matrix_t b = zeros<value_t>(1, hiddenCount);
    layer.set_weights(W);
    layer.set_bias(b);

    matrix_t means = zeros<value_t>(1, visibleCount);
    matrix_t stddev = ones<value_t>(1, visibleCount);

    if (getNormalizeInputs() && iLayer == 0) {

      matrix_t X(sampleCount, visibleCount);
      for (size_t i = 0; i < sampleCount; ++i) {
        thrust::copy(data[i]->begin(), data[i]->end(), row(X, i).begin());
      }

      means = ones<value_t>(means.size()) * sum(X) / X.count();
  //    means = sum(X, 0);
  //    means = means / X.size()[0];
      X = X - repeat(means, X.size() / means.size());

      stddev = ones<value_t>(stddev.size()) * sqrt(dot(X, X) / X.count());
  //    matrix_t temp = X * X;
  //    stddev = sum(temp, 0);
  //    stddev = sqrt(stddev / X.size()[0]) + (stddev == 0);  // If stddev == 0 set stddev to 1
    }

    layer.set_mean(means);
    layer.set_stddev(stddev);

    model->append_layer(layer);
  }

  newState->setModel(model);
}

} /* namespace nn */

} /* namespace gml */
