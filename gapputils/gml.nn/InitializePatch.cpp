/*
 * InitializePatch.cpp
 *
 *  Created on: Aug 13, 2014
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

namespace nn {

BeginPropertyDefinitions(InitializePatch)

  ReflectableBase(DefaultWorkflowElement<InitializePatch>)

  WorkflowProperty(TrainingSet, Input("D"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Labels, Input("L"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(PatchWidth)
  WorkflowProperty(PatchHeight)
  WorkflowProperty(PatchDepth)
  WorkflowProperty(HiddenUnitCounts)
  WorkflowProperty(InitialWeights)
  WorkflowProperty(HiddenActivationFunction, Enumerator<Type>())
  WorkflowProperty(OutputActivationFunction, Enumerator<Type>())
  WorkflowProperty(NormalizeInputs, Flag())
  WorkflowProperty(Model, Output("NN"))

EndPropertyDefinitions

InitializePatch::InitializePatch() : _PatchWidth(16), _PatchHeight(16), _PatchDepth(16), _InitialWeights(0.001), _NormalizeInputs(true) {
  setLabel("Init");
}

void InitializePatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef float value_t;
  typedef tbblas::tensor<value_t, 2> matrix_t;
  typedef random_tensor<value_t, 2, false, normal<value_t> > randn_t;
  typedef tensor_t::dim_t dim_t;

  v_tensor_t& tensors = *getTrainingSet();
  v_tensor_t& labels = *getLabels();


  const int dimCount = tensor_t::dimCount;
  for (int i = 0; i < dimCount - 1; ++i) {
    if (tensors[0]->size()[i] != labels[0]->size()[i]) {
      dlog(Severity::Warning) << "The input tensors and the labels must have the same width, height, and depth. Aborting!";
      return;
    }
  }

  const size_t sampleCount = tensors.size();

  if (sampleCount != labels.size()) {
    dlog(Severity::Warning) << "The number of samples and labels must be the same. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t());
  const std::vector<int>& hiddens = getHiddenUnitCounts();

  for (size_t iLayer = 0; iLayer < hiddens.size() + 1; ++iLayer) {
    const size_t visibleCount = iLayer == 0 ? getPatchWidth() * getPatchHeight() * getPatchDepth() * tensors[0]->size()[dimCount - 1] : hiddens[iLayer - 1];
    const size_t hiddenCount = iLayer == hiddens.size() ? labels[0]->size()[dimCount - 1] : hiddens[iLayer];

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

      // normalize channels individually
      int channelCount = tensors[0]->size()[dimCount - 1];
      int channelLength = tensors[0]->count() / channelCount;
      int patchLength = visibleCount / channelCount;
      dim_t channelSize = tensors[0]->size();
      channelSize[dimCount - 1] = 1;

      tensor_t channel;

      for (int iChannel = 0; iChannel < channelCount; ++iChannel) {
        value_t mean = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
          mean += sum((*tensors[i])[seq(0,0,0,iChannel), channelSize]) / channelLength;
        }
        mean /= tensors.size();

        dlog(Severity::Trace) << "Mean of channel " << iChannel + 1 << ": " << mean;

        value_t sd = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
          channel = (*tensors[i])[seq(0,0,0,iChannel), channelSize];
          sd += dot(channel - mean, channel - mean) / channelLength;
        }
        dlog(Severity::Trace) << "Standard deviation of channel " << iChannel + 1 << ": " << sqrt(sd / tensors.size());

        means[seq(0, patchLength * iChannel), seq(1, patchLength)] = ones<value_t>(1, patchLength) * mean;
        stddev[seq(0, patchLength * iChannel), seq(1, patchLength)] = ones<value_t>(1, patchLength) * sqrt(sd / tensors.size());
      }
    }

    layer.set_mean(means);
    layer.set_stddev(stddev);

    model->append_layer(layer);
  }

  newState->setModel(model);
}

} /* namespace nn */

} /* namespace gml */
