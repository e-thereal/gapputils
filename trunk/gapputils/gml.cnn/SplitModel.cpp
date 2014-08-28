/*
 * SplitModel.cpp
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "SplitModel.h"

#include <tbblas/linalg.hpp>

#include <algorithm>

namespace gml {

namespace cnn {

BeginPropertyDefinitions(SplitModel)

  ReflectableBase(DefaultWorkflowElement<SplitModel>)

  WorkflowProperty(Model, Input("CNN"), NotNull<Type>())
  WorkflowProperty(Layer)
  WorkflowProperty(Filters, Output("F"))
  WorkflowProperty(Biases, Output("Bs"))
  WorkflowProperty(Weights, Output("W"))
  WorkflowProperty(Bias, Output("B"))

EndPropertyDefinitions

SplitModel::SplitModel() : _Layer(0) {
  setLabel("Split");
}

void SplitModel::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  int clayerCount = getModel()->cnn_layers().size();
  int dlayerCount = getModel()->nn_layers().size();

  if (getLayer() >= clayerCount + dlayerCount) {
    dlog(Severity::Warning) << "Invalid layer specified. The given network has only " << clayerCount + dlayerCount << " layers. Aborting!";
    return;
  }

  if (getLayer() < clayerCount) {
    cnn_layer_t& cnn_layer = *getModel()->cnn_layers()[getLayer()];

    boost::shared_ptr<v_tensor_t> filters(new v_tensor_t());
    for (size_t i = 0; i < cnn_layer.filters().size(); ++i)
      filters->push_back(boost::make_shared<tensor_t>(*cnn_layer.filters()[i]));
    newState->setFilters(filters);

    boost::shared_ptr<v_tensor_t> bias(new v_tensor_t());
    for (size_t i = 0; i < cnn_layer.bias().size(); ++i)
      bias->push_back(boost::make_shared<tensor_t>(*cnn_layer.bias()[i]));
    newState->setBiases(bias);

  } else {
    model_t::nn_layer_t& nn_layer = *getModel()->nn_layers()[getLayer() - clayerCount];

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
  }
}

} /* namespace nn */

} /* namespace gml */
