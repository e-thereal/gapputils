/*
 * SplitModel.cpp
 *
 *  Created on: Jan 05, 2015
 *      Author: tombr
 */

#include "SplitModel.h"

#include <tbblas/linalg.hpp>

#include <algorithm>

namespace gml {

namespace encoder {

BeginPropertyDefinitions(SplitModel)

  ReflectableBase(DefaultWorkflowElement<SplitModel>)

  WorkflowProperty(Model, Input("ENN"), NotNull<Type>())
  WorkflowProperty(Layer)
  WorkflowProperty(Shortcut, Flag())
  WorkflowProperty(Filters, Output("F"))
  WorkflowProperty(Biases, Output("Bs"))
  WorkflowProperty(Weights, Output("W"))
  WorkflowProperty(Bias, Output("B"))

EndPropertyDefinitions

SplitModel::SplitModel() : _Layer(0), _Shortcut(false) {
  setLabel("Split");
}

void SplitModel::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  int celayerCount = getModel()->cnn_encoders().size();
  int delayerCount = getModel()->nn_encoders().size();
  int cdlayerCount = getModel()->dnn_decoders().size();
  int ddlayerCount = getModel()->nn_decoders().size();

  if (getLayer() >= celayerCount + delayerCount + cdlayerCount + ddlayerCount) {
    dlog(Severity::Warning) << "Invalid layer specified. The given network has only " << celayerCount + delayerCount + cdlayerCount + ddlayerCount << " layer(s). Aborting!";
    return;
  }

  if (getLayer() < celayerCount) {
    dlog(Severity::Trace) << "Getting information from convolutional encoding layer " << getLayer();
    cnn_layer_t& cnn_layer = *getModel()->cnn_encoders()[getLayer()];

    boost::shared_ptr<v_tensor_t> filters(new v_tensor_t());
    for (size_t i = 0; i < cnn_layer.filters().size(); ++i)
      filters->push_back(boost::make_shared<tensor_t>(*cnn_layer.filters()[i]));
    newState->setFilters(filters);

    boost::shared_ptr<v_tensor_t> bias(new v_tensor_t());
    for (size_t i = 0; i < cnn_layer.bias().size(); ++i)
      bias->push_back(boost::make_shared<tensor_t>(*cnn_layer.bias()[i]));
    newState->setBiases(bias);

  } else if (getLayer() < celayerCount + delayerCount) {
    dlog(Severity::Trace) << "Getting information from dense encoding layer " << getLayer() - celayerCount;
    model_t::nn_layer_t& nn_layer = *getModel()->nn_encoders()[getLayer() - celayerCount];

    typedef model_t::nn_layer_t::host_matrix_t matrix_t;

    matrix_t W = nn_layer.weights();
    const matrix_t& b = nn_layer.bias();

    boost::shared_ptr<v_data_t> weights(new v_data_t());
    boost::shared_ptr<data_t> bias(new data_t(b.count()));

    for (int iCol = 0; iCol < W.size()[1]; ++iCol) {
      boost::shared_ptr<data_t> col(new data_t(W.size()[0]));
      std::copy(column(W, iCol).begin(), column(W, iCol).end(), col->begin());
      weights->push_back(col);
    }
    std::copy(b.begin(), b.end(), bias->begin());

    newState->setWeights(weights);
    newState->setBias(bias);
  } else if (getLayer() < celayerCount + delayerCount + ddlayerCount) {
    dlog(Severity::Trace) << "Getting information from dense decoding layer " << getLayer() - celayerCount - delayerCount;
    model_t::nn_layer_t& nn_layer = *getModel()->nn_decoders()[getLayer() - celayerCount - delayerCount];

    typedef model_t::nn_layer_t::host_matrix_t matrix_t;

    matrix_t W = nn_layer.weights();
    const matrix_t& b = nn_layer.bias();

    boost::shared_ptr<v_data_t> weights(new v_data_t());
    boost::shared_ptr<data_t> bias(new data_t(b.count()));

    for (int iCol = 0; iCol < W.size()[1]; ++iCol) {
      boost::shared_ptr<data_t> col(new data_t(W.size()[0]));
      std::copy(column(W, iCol).begin(), column(W, iCol).end(), col->begin());
      weights->push_back(col);
    }
    std::copy(b.begin(), b.end(), bias->begin());

    newState->setWeights(weights);
    newState->setBias(bias);
  } else if (getLayer() < celayerCount + delayerCount + ddlayerCount + cdlayerCount) {
    const int iLayer = getLayer() - celayerCount - delayerCount - ddlayerCount;
    dlog(Severity::Trace) << "Getting information from convolutional decoding layer " << iLayer;
    if (!_Shortcut) {
      dnn_layer_t& cnn_layer = *getModel()->dnn_decoders()[iLayer];

      boost::shared_ptr<v_tensor_t> filters(new v_tensor_t());
      for (size_t i = 0; i < cnn_layer.filters().size(); ++i)
        filters->push_back(boost::make_shared<tensor_t>(*cnn_layer.filters()[i]));
      newState->setFilters(filters);

      boost::shared_ptr<v_tensor_t> bias(new v_tensor_t(1));
      bias->at(0) = boost::make_shared<tensor_t>(cnn_layer.bias());
      newState->setBiases(bias);
    } else if (iLayer > 0) {
      dnn_layer_t& cnn_layer = *getModel()->dnn_shortcuts()[iLayer - 1];

      boost::shared_ptr<v_tensor_t> filters(new v_tensor_t());
      for (size_t i = 0; i < cnn_layer.filters().size(); ++i)
        filters->push_back(boost::make_shared<tensor_t>(*cnn_layer.filters()[i]));
      newState->setFilters(filters);

      boost::shared_ptr<v_tensor_t> bias(new v_tensor_t(1));
      bias->at(0) = boost::make_shared<tensor_t>(cnn_layer.bias());
      newState->setBiases(bias);
    } else {
      dlog(Severity::Message) << "Given decoding layer does not have a short cut connection.";
    }
  }
}

} /* namespace encoders */

} /* namespace gml */
