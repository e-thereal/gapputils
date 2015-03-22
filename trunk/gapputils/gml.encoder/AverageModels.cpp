/*
 * AverageModels.cpp
 *
 *  Created on: Jan 30, 2015
 *      Author: tombr
 */

#include "AverageModels.h"

namespace gml {

namespace encoder {

BeginPropertyDefinitions(AverageModels)

  ReflectableBase(DefaultWorkflowElement<AverageModels>)

  WorkflowProperty(Model1, Input("ENN1"), NotNull<Type>())
  WorkflowProperty(Model2, Input("ENN2"), NotNull<Type>())
  WorkflowProperty(AverageModel, Output("AVG"))

EndPropertyDefinitions

AverageModels::AverageModels() {
  setLabel("AvgModel");
}

void AverageModels::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  typedef model_t::cnn_layer_t cnn_layer_t;
  typedef model_t::dnn_layer_t dnn_layer_t;

  typedef model_t::v_cnn_layer_t v_cnn_layer_t;
  typedef model_t::v_dnn_layer_t v_dnn_layer_t;

  model_t& model1 = *getModel1();
  model_t& model2 = *getModel2();

  const int celayerCount = model1.cnn_encoders().size();
  const int cdlayerCount = model1.dnn_decoders().size();

  if (model1.nn_encoders().size() || model1.nn_decoders().size() ||
      model2.nn_encoders().size() || model2.nn_decoders().size())
  {
    dlog(Severity::Warning) << "This module does not yet support averaging dense layers. Aborting!";
    return;
  }

  if ((int)model2.cnn_encoders().size() != celayerCount || (int)model2.dnn_decoders().size() != cdlayerCount) {
    dlog(Severity::Warning) << "The two input models must have the same number of layers. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> avgModel = boost::make_shared<model_t>(model1);

  // Go through all layers and average each layer
  for (int iLayer = 0; iLayer < celayerCount; ++iLayer) {
    cnn_layer_t& encoder1 = *avgModel->cnn_encoders()[iLayer];
    cnn_layer_t& encoder2 = *model2.cnn_encoders()[iLayer];

    // Check compatibility
    assert(encoder1.version() == encoder2.version());
    assert(encoder1.kernel_size() == encoder2.kernel_size());
    assert(encoder1.stride_size() == encoder2.stride_size());
    assert(encoder1.pooling_size() == encoder2.pooling_size());
    assert(encoder1.visibles_size() == encoder2.visibles_size());
    assert(encoder1.activation_function() == encoder2.activation_function());
    assert(encoder1.convolution_type() == encoder2.convolution_type());
    encoder1.set_mean(0.5 * (encoder1.mean() + encoder2.mean()));
    encoder1.set_stddev(0.5 * (encoder1.stddev() + encoder2.stddev()));
    assert(encoder1.shared_bias() == encoder2.shared_bias());

    assert(encoder1.filters().size() == encoder2.filters().size());
    for (size_t iFilter = 0; iFilter < encoder1.filters().size(); ++iFilter) {
      *encoder1.filters()[iFilter] = 0.5 * (*encoder1.filters()[iFilter] + *encoder2.filters()[iFilter]);
    }

    assert(encoder1.bias().size() == encoder2.bias().size());
    for (size_t iBias = 0; iBias < encoder1.bias().size(); ++iBias) {
      *encoder1.bias()[iBias] = 0.5 * (*encoder1.bias()[iBias] + *encoder2.bias()[iBias]);
    }
  }

  for (int iLayer = 0; iLayer < cdlayerCount; ++iLayer) {
    dnn_layer_t& decoder1 = *avgModel->dnn_decoders()[iLayer];
    dnn_layer_t& decoder2 = *model2.dnn_decoders()[iLayer];

    // Check compatibility
    assert(decoder1.version() == decoder2.version());
    assert(decoder1.kernel_size() == decoder2.kernel_size());
    assert(decoder1.stride_size() == decoder2.stride_size());
    assert(decoder1.pooling_size() == decoder2.pooling_size());
    assert(decoder1.visibles_size() == decoder2.visibles_size());
    assert(decoder1.activation_function() == decoder2.activation_function());
    assert(decoder1.convolution_type() == decoder2.convolution_type());
    decoder1.set_mean(0.5 * (decoder1.mean() + decoder2.mean()));
    decoder1.set_stddev(0.5 * (decoder1.stddev() + decoder2.stddev()));
    assert(decoder1.shared_bias() == decoder2.shared_bias());

    assert(decoder1.filters().size() == decoder2.filters().size());
    for (size_t iFilter = 0; iFilter < decoder1.filters().size(); ++iFilter) {
      *decoder1.filters()[iFilter] = 0.5 * (*decoder1.filters()[iFilter] + *decoder2.filters()[iFilter]);
    }

    decoder1.set_bias(0.5 * (decoder1.bias() + decoder2.bias()));
    assert(sum(decoder1.mask() == decoder2.mask()) == decoder1.mask().count());
  }

  newState->setAverageModel(avgModel);
}

} /* namespace encoder */

} /* namespace gml */
