/*
 * SplitModel.cpp
 *
 *  Created on: Dec 12, 2013
 *      Author: tombr
 */

#include "SplitModel.h"

#include <tbblas/linalg.hpp>

#include <algorithm>

namespace gml {

namespace rbm {

BeginPropertyDefinitions(SplitModel)

  ReflectableBase(DefaultWorkflowElement<SplitModel>)

  WorkflowProperty(Model, Input("RBM"), NotNull<Type>())
  WorkflowProperty(Weights, Output("W"))
  WorkflowProperty(VisibleBias, Output("VB"))
  WorkflowProperty(HiddenBias, Output("HB"))

EndPropertyDefinitions

SplitModel::SplitModel() {
  setLabel("SplitModel");
}

void SplitModel::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  model_t& rbm = *getModel();

  typedef model_t::host_matrix_t matrix_t;

  matrix_t W = rbm.weights();
  const matrix_t& b = rbm.visible_bias();
  const matrix_t& c = rbm.hidden_bias();

  boost::shared_ptr<v_data_t> weights(new v_data_t());
  boost::shared_ptr<data_t> vb(new data_t(b.count()));
  boost::shared_ptr<data_t> hb(new data_t(c.count()));

  for (int iCol = 0; iCol < W.size()[1]; ++iCol) {
    boost::shared_ptr<data_t> col(new data_t(W.size()[0]));
    std::copy(column(W, iCol).begin(), column(W, iCol).end(), col->begin());
    weights->push_back(col);
  }
  std::copy(b.begin(), b.end(), vb->begin());
  std::copy(c.begin(), c.end(), hb->begin());

  newState->setWeights(weights);
  newState->setVisibleBias(vb);
  newState->setHiddenBias(hb);
}

} /* namespace rbm */

} /* namespace gml */
