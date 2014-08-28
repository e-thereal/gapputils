/*
 * BackpropWeights_gpu.cu
 *
 *  Created on: 2014-08-16
 *      Author: tombr
 */

#include "BackpropWeights.h"

#include <tbblas/deeplearn/nn_layer.hpp>
#include <tbblas/deeplearn/cnn_layer.hpp>
#include <tbblas/identity.hpp>
#include <tbblas/util.hpp>

#include <tbblas/rearrange.hpp>

#include <tbblas/io.hpp>

namespace gml {

namespace cnn {

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

  typedef tensor<value_t, dimCount, true> tensor_t;

//  typedef model_t::nn_layer_t::host_matrix_t host_matrix_t;

  typedef tbblas::deeplearn::nn_layer<value_t> nn_layer_t;
  typedef tbblas::deeplearn::cnn_layer<value_t, dimCount> cnn_layer_t;

  model_t& model = *getModel();
  size_t dlayerCount = model.nn_layers().size();
  size_t clayerCount = model.cnn_layers().size();

  if (getLayer() >= dlayerCount) {
    dlog(Severity::Warning) << "Invalid layer specified. The given network has only " << dlayerCount << " dense layers. Aborting!";
    return;
  }

  matrix_t dW = identity<value_t>(model.nn_layers()[getLayer()]->hiddens_count());
  for (int iLayer = getLayer(); iLayer >= 0; --iLayer) {
    nn_layer_t layer(*model.nn_layers()[iLayer]);
    layer.hiddens() = dW;
    layer.backprop_visibles();
    dW = layer.visibles();
  }

  boost::shared_ptr<v_host_tensor_t> weights(new v_host_tensor_t());
  for (int iRow = 0; iRow < dW.size()[0]; ++iRow) {
    tensor_t cW(model.cnn_layers()[clayerCount - 1]->hiddens_size());
    std::copy(row(dW, iRow).begin(), row(dW, iRow).end(), cW.begin());

    for (int iLayer = clayerCount - 1; iLayer >= 0; --iLayer) {
      cnn_layer_t layer(*model.cnn_layers()[iLayer]);
      layer.hiddens() = cW;
      layer.backprop_visibles();
      cW = rearrange_r(layer.visibles(), model.cnn_layers()[iLayer]->stride_size());
    }

    weights->push_back(boost::make_shared<host_tensor_t>(cW));
  }
  tbblas::synchronize();

  newState->setWeights(weights);
}

}

}
