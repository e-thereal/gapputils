/*
 * CreateDbm_gpu.cu
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "CreateDbm.h"

#include <tbblas/rearrange.hpp>

namespace gml {

namespace convrbm4d {

CreateDbmChecker::CreateDbmChecker() {
  CreateDbm test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Dataset, test);
  CHECK_MEMORY_LAYOUT2(CrbmModels, test);
  CHECK_MEMORY_LAYOUT2(DbmModel, test);
}

void CreateDbm::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef host_tensor_t::dim_t dim_t;
  const unsigned dimCount = host_tensor_t::dimCount;

  boost::shared_ptr<DbmModel> model(new DbmModel());

  // Initialize DBM from several RBMs
  // - Copy filters
  // - Average biases of intermediate layers
  // - Copy masks

  boost::shared_ptr<vv_host_tensor_t> weights(new vv_host_tensor_t());
  boost::shared_ptr<vv_host_tensor_t> hiddenBiases(new vv_host_tensor_t());
  boost::shared_ptr<v_host_tensor_t> masks(new v_host_tensor_t());

  std::vector<boost::shared_ptr<Model> >& crbms = *getCrbmModels();
  for (size_t iLayer = 0; iLayer < crbms.size(); ++iLayer) {

    // Copy filters
    v_host_tensor_t& oldFilters = *crbms[iLayer]->getFilters();
    boost::shared_ptr<v_host_tensor_t> newFilters(new v_host_tensor_t());
    for (size_t iFilter = 0; iFilter < oldFilters.size(); ++iFilter)
      newFilters->push_back(boost::make_shared<host_tensor_t>(*oldFilters[iFilter]));
    weights->push_back(newFilters);

    // Copy biases
    v_host_tensor_t& oldBiases = *crbms[iLayer]->getHiddenBiases();
    boost::shared_ptr<v_host_tensor_t> newBiases(new v_host_tensor_t());
    for (size_t iFilter = 0; iFilter < oldFilters.size(); ++iFilter) {
      boost::shared_ptr<host_tensor_t> bias(new host_tensor_t(*oldBiases[iFilter]));
      if (iLayer + 1 < crbms.size()) {
        host_tensor_t& vbias = *crbms[iLayer + 1]->getVisibleBias();
        dim_t offset(0), sliceDim = vbias.size(), blockSize = oldBiases[0]->size() / vbias.size();
        offset[dimCount - 1] = sliceDim[dimCount - 1] =
            vbias.size()[dimCount - 1] / oldBiases[0]->size()[dimCount - 1] / oldFilters.size();
        blockSize[dimCount - 1] = 1;
        *bias = (*bias + rearrange_r(vbias[offset * iFilter, sliceDim], blockSize)) / 2.0;
      }
      newBiases->push_back(bias);
    }
    hiddenBiases->push_back(newBiases);

    // Copy mask
    masks->push_back(boost::make_shared<host_tensor_t>(*crbms[iLayer]->getMask()));
  }
  model->setWeights(weights);
  model->setVisibleBias(boost::make_shared<host_tensor_t>(*crbms[0]->getVisibleBias()));
  model->setHiddenBiases(hiddenBiases);
  model->setMasks(masks);

  dim_t visibleBlock = getDataset()->at(0)->size() / hiddenBiases->at(0)->at(0)->size();
  visibleBlock[dimCount - 1] = 1;
  model->setVisibleBlockSize(visibleBlock);
  model->setMean(crbms[0]->getMean());
  model->setStddev(crbms[0]->getStddev());

  newState->setDbmModel(model);
}

}

}
