/*
 * VisualizeFilters_gpu.cu
 *
 *  Created on: Aug 6, 2015
 *      Author: tombr
 */

#include "VisualizeFilters.h"

#include <tbblas/deeplearn/encoder.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/util.hpp>

#include <tbblas/io.hpp>
#include <tbblas/dot.hpp>

#include <boost/ref.hpp>

namespace gml {

namespace encoder {

void VisualizeFilters::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef model_t::value_t value_t;
  const unsigned dimCount = model_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  model_t& model = *getModel();

  // Get layer Id
  int maxLayer = getLayer() >= 0 ? min(getLayer(), (int)model.cnn_encoders().size() - 1) : model.cnn_encoders().size() - 1;

  tensor_t paddedFilter;
  host_tensor_t filter;

  // Visualize encoders
  {
    boost::shared_ptr<v_host_tensor_t> filters(new v_host_tensor_t());
    cnn_layer_t& lastCnn = *model.cnn_encoders()[maxLayer];
    std::vector<boost::shared_ptr<tbblas::deeplearn::cnn_layer<value_t, dimCount> > > layers;
    for (size_t iLayer = 0; iLayer <= maxLayer; ++iLayer) {
      layers.push_back(boost::make_shared<tbblas::deeplearn::cnn_layer<value_t, dimCount> >(boost::ref(*model.cnn_encoders()[iLayer])));
    }

    for (size_t iFilter = 0; iFilter < lastCnn.filter_count(); ++iFilter) {
      dim_t filterSize = seq<dimCount>(1);
      dim_t topleft = lastCnn.outputs_size() / 2;
      topleft[dimCount - 1] = iFilter;

      paddedFilter = zeros<value_t>(lastCnn.outputs_size());
      paddedFilter[topleft] = 1.0;

      layers[maxLayer]->hiddens() = paddedFilter;

      // Manually perform inference.
      for (int iLayer = maxLayer; iLayer >= 0; --iLayer) {
        filterSize = (filterSize * model.cnn_encoders()[iLayer]->pooling_size() - 1) * model.cnn_encoders()[iLayer]->stride_size() + 1 + model.cnn_encoders()[iLayer]->kernel_size() - 1;
        topleft = topleft * model.cnn_encoders()[iLayer]->pooling_size() * model.cnn_encoders()[iLayer]->stride_size();

        layers[iLayer]->backprop_visibles();
        if (iLayer > 0) {
          layers[iLayer - 1]->hiddens() = layers[iLayer]->visibles();
        }
      }

      topleft[dimCount - 1] = 0;
      filterSize[dimCount - 1] = model.cnn_encoders()[0]->visibles_size()[dimCount - 1];

      filter = layers[0]->visibles();
      tbblas::synchronize();
      boost::shared_ptr<host_tensor_t> h_filter = boost::make_shared<host_tensor_t>(filter[topleft, filterSize]);
      filters->push_back(h_filter);

      if (monitor)
        monitor->reportProgress(100. * (iFilter + 1) / (2 * lastCnn.filter_count()));
    }
    newState->setEncodingFilters(filters);
  }

  // Visualize Decoders
  {
    boost::shared_ptr<v_host_tensor_t> filters(new v_host_tensor_t());
    dnn_layer_t& lastDnn = *model.dnn_decoders()[model.dnn_decoders().size() - maxLayer - 1];
    std::vector<boost::shared_ptr<tbblas::deeplearn::dnn_layer<value_t, dimCount> > > layers;
    for (size_t iLayer = 0; iLayer <= maxLayer; ++iLayer) {
      tbblas_print(iLayer + model.dnn_decoders().size() - maxLayer - 1);
      layers.push_back(boost::make_shared<tbblas::deeplearn::dnn_layer<value_t, dimCount> >(boost::ref(*model.dnn_decoders()[iLayer + model.dnn_decoders().size() - maxLayer - 1])));
    }

    for (size_t iFilter = 0; iFilter < lastDnn.filter_count(); ++iFilter) {
      dim_t filterSize = seq<dimCount>(1);
      dim_t topleft = lastDnn.outputs_size() / 2;
      topleft[dimCount - 1] = iFilter;

      paddedFilter = zeros<value_t>(lastDnn.outputs_size());
      paddedFilter[topleft] = 1.0;

      layers[0]->hiddens() = paddedFilter;

      // Manually perform inference.
      for (int iLayer = 0; iLayer <= maxLayer; ++iLayer) {
        const int mLayer = iLayer + model.dnn_decoders().size() - maxLayer - 1;

        // TODO: adjust this calculation based on visible_pooling or not
        filterSize = (filterSize * model.cnn_encoders()[mLayer]->pooling_size() - 1) * model.cnn_encoders()[mLayer]->stride_size() + 1 + model.cnn_encoders()[mLayer]->kernel_size() - 1;
        topleft = topleft * model.cnn_encoders()[mLayer]->pooling_size() * model.cnn_encoders()[mLayer]->stride_size();

        layers[iLayer]->infer_visibles(tbblas::deeplearn::dnn_layer<value_t, dimCount>::APPLY_BIAS);
        if (iLayer < maxLayer) {
          layers[iLayer + 1]->hiddens() = layers[iLayer]->visibles();
        }
      }

      topleft[dimCount - 1] = 0;
      filterSize[dimCount - 1] = model.dnn_decoders()[model.dnn_decoders().size() - 1]->visibles_size()[dimCount - 1];

      filter = layers[maxLayer]->visibles();
      tbblas::synchronize();
      boost::shared_ptr<host_tensor_t> h_filter = boost::make_shared<host_tensor_t>(filter[topleft, filterSize]);
//      boost::shared_ptr<host_tensor_t> h_filter = boost::make_shared<host_tensor_t>(filter);
      filters->push_back(h_filter);

      if (monitor)
        monitor->reportProgress(100. * (iFilter + 1 + lastDnn.filter_count()) / (2 * lastDnn.filter_count()));
    }
    newState->setDecodingFilters(filters);
  }
}

}

}
