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

namespace nn {

BeginPropertyDefinitions(SplitModel)

  ReflectableBase(DefaultWorkflowElement<SplitModel>)

  WorkflowProperty(Model, Input("NN"), NotNull<Type>())
  WorkflowProperty(Weights, Output("W"))
  WorkflowProperty(Bias, Output("B"))

EndPropertyDefinitions

SplitModel::SplitModel() {
  setLabel("Split");
}

void SplitModel::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  nn_layer_t& nn_layer = *getModel();

  typedef nn_layer_t::host_matrix_t matrix_t;

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

} /* namespace nn */

} /* namespace gml */
