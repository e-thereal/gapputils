/*
 * VisualizeWeights_gpu.cu
 *
 *  Created on: Aug 29, 2014
 *      Author: tombr
 */

#include "VisualizeWeights.h"

#include <tbblas/deeplearn/dbn.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/util.hpp>
#include <tbblas/linalg.hpp>

#include <tbblas/io.hpp>
#include <tbblas/dot.hpp>

//#include <tbblas/new_context.hpp>
//#include <tbblas/change_stream.hpp>

namespace gml {

namespace dbn {

VisualizeWeightsChecker::VisualizeWeightsChecker() {
  VisualizeWeights test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchLength, test);
  CHECK_MEMORY_LAYOUT2(Weights, test);
}

void VisualizeWeights::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef dbn_t::value_t value_t;
  const unsigned dimCount = dbn_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor<value_t, 2, true> matrix_t;
  typedef tensor_t::dim_t dim_t;

  boost::shared_ptr<v_host_tensor_t> weights(new v_host_tensor_t());

  dbn_t& model = *getModel();
  tbblas::deeplearn::dbn<value_t, dimCount> dbn(model);
  for (size_t i = 0; i < getModel()->crbms().size() && i < getFilterBatchLength().size(); ++i)
    dbn.set_batch_length(i, getFilterBatchLength()[i]);

  if (model.crbms().size() < 1) {
    dlog(Severity::Warning) << "At least one convolutional layer required. Aborting!";
    return;
  }

  if (model.rbms().size()) {
    // Reconstruct from dense layers (no cropping required)
    matrix_t ident = identity<value_t>(model.rbms()[model.rbms().size() - 1]->hiddens_count());
    for (size_t iRow = 0; iRow < ident.size()[0]; ++iRow) {
      dbn.hiddens() = row(ident, iRow);
      dbn.infer_visibles(-1, true);
      weights->push_back(boost::make_shared<host_tensor_t>(dbn.cvisibles()));
      tbblas::synchronize();
    }
  } else {
    // Reconstruct from convolutional layers (requires cropping)

    crbm_t& lastCrbm = *model.crbms()[model.crbms().size() - 1];
    tensor_t paddedFilter, filter;

    for (size_t i = 0; i < lastCrbm.filter_count(); ++i) {
      dim_t filterSize = seq<dimCount>(1);
      dim_t topleft = lastCrbm.hiddens_size() / 2;
      topleft[dimCount - 1] = i;

      paddedFilter = zeros<value_t>(lastCrbm.hiddens_size());
      paddedFilter[topleft] = 1.0f;

      dbn.chiddens() = paddedFilter;
      dbn.infer_visibles(-1, true);

      for (int iLayer = model.crbms().size() - 1; iLayer >= 0; --iLayer) {
        filterSize = filterSize + model.crbms()[iLayer]->kernel_size() - 1;

        if (iLayer > 0) {
          filterSize = filterSize * model.stride_size(iLayer);
          topleft = topleft * model.stride_size(iLayer);
        }
      }

      topleft[dimCount - 1] = 0;
      filterSize[dimCount - 1] = model.crbms()[0]->visibles_size()[dimCount - 1];
      filter = dbn.cvisibles()[topleft, filterSize];
      weights->push_back(boost::make_shared<host_tensor_t>(filter));
      tbblas::synchronize();
    }
  }

  newState->setWeights(weights);
}

}

}
