/*
 * BackpropWeights_gpu.cu
 *
 *  Created on: 2014-08-16
 *      Author: tombr
 */

#include "BackpropWeights.h"

#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/identity.hpp>
#include <tbblas/util.hpp>

#include <tbblas/io.hpp>

namespace gml {

namespace nn {

BackpropWeightsChecker::BackpropWeightsChecker() {
  BackpropWeights test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Layer, test);
  CHECK_MEMORY_LAYOUT2(Weights, test);
}

void BackpropWeights::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef model_t::value_t value_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef model_t::nn_layer_t::host_matrix_t host_matrix_t;
  typedef tbblas::deeplearn::nn_layer<value_t> layer_t;

  model_t& model = *getModel();
  size_t layerCount = model.layers().size();

  if (getLayer() >= layerCount) {
    dlog(Severity::Warning) << "Invalid layer specified. The given network has only " << layerCount << " layers. Aborting!";;
    return;
  }

  matrix_t dW = identity<value_t>(model.layers()[getLayer()]->hiddens_count());
  for (int iLayer = getLayer(); iLayer >= 0; --iLayer) {
    layer_t layer(*model.layers()[iLayer]);
    layer.hiddens() = dW;
    layer.backprop_visibles();
    dW = layer.visibles();
  }

  host_matrix_t W = dW;
  tbblas::synchronize();

  boost::shared_ptr<v_data_t> weights(new v_data_t());
  for (int iRow = 0; iRow < W.size()[0]; ++iRow) {
    boost::shared_ptr<data_t> data(new data_t(W.size()[1]));
    std::copy(row(W, iRow).begin(), row(W, iRow).end(), data->begin());
    weights->push_back(data);
  }

  newState->setWeights(weights);
}

}

}
