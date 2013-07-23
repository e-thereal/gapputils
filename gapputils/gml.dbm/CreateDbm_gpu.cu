/*
 * CreateDbm_gpu.cu
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "CreateDbm.h"

#include <tbblas/rearrange.hpp>

namespace gml {

namespace dbm {

CreateDbmChecker::CreateDbmChecker() {
  CreateDbm test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Dataset, test);
  CHECK_MEMORY_LAYOUT2(CrbmModels, test);
  CHECK_MEMORY_LAYOUT2(RbmModels, test);
  CHECK_MEMORY_LAYOUT2(DbmModel, test);
}

void CreateDbm::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef host_tensor_t::dim_t dim_t;
  const unsigned dimCount = host_tensor_t::dimCount;

  boost::shared_ptr<Model> model(new Model());

  // Initialize DBM from several RBMs
  // - Copy filters
  // - Average biases of intermediate layers
  // - Copy masks

  boost::shared_ptr<vv_host_tensor_t> weights(new vv_host_tensor_t());
  boost::shared_ptr<vv_host_tensor_t> hiddenBiases(new vv_host_tensor_t());
  boost::shared_ptr<v_host_tensor_t> masks(new v_host_tensor_t());
  boost::shared_ptr<v_host_matrix_t> matrices(new v_host_matrix_t());
  boost::shared_ptr<v_host_matrix_t> flatBiases(new v_host_matrix_t());

  std::vector<boost::shared_ptr<gml::convrbm4d::Model> >& crbms = *getCrbmModels();
  std::vector<boost::shared_ptr<gml::rbm::Model> >& rbms = *getRbmModels();

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
      if (iLayer < crbms.size() - 1) {
        host_tensor_t& vbias = *crbms[iLayer + 1]->getVisibleBias();
        dim_t offset(0), sliceDim = vbias.size(), blockSize = oldBiases[0]->size() / vbias.size();
        offset[dimCount - 1] = sliceDim[dimCount - 1] =
            vbias.size()[dimCount - 1] / oldBiases[0]->size()[dimCount - 1] / oldFilters.size();
        blockSize[dimCount - 1] = 1;
        *bias = (*bias + rearrange_r(vbias[offset * iFilter, sliceDim], blockSize)) / 2.0;
      } else {
        host_tensor_t vbias(bias->size());
        assert(rbms[0]->getVisibleBiases()->count() == vbias.count() * oldFilters.size());
        thrust::copy(rbms[0]->getVisibleBiases()->begin() + iFilter * vbias.count(),
            rbms[0]->getVisibleBiases()->begin() + (iFilter + 1) * vbias.count(), vbias.begin());
        *bias = (*bias + vbias) / 2.0;
      }
      newBiases->push_back(bias);
    }
    hiddenBiases->push_back(newBiases);

    // Copy mask
    masks->push_back(boost::make_shared<host_tensor_t>(*crbms[iLayer]->getMask()));
  }

  for (size_t iLayer = 0; iLayer < rbms.size(); ++iLayer) {
    boost::shared_ptr<host_matrix_t> bias(new host_matrix_t(*rbms[iLayer]->getHiddenBiases()));
    if (iLayer < rbms.size() - 1) {
      *bias = (*bias + *rbms[iLayer + 1]->getVisibleBiases()) / 2.0;
    }
    flatBiases->push_back(bias);
    matrices->push_back(boost::make_shared<host_matrix_t>(*rbms[iLayer]->getWeightMatrix()));
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

  model->setWeightMatrices(matrices);
  model->setFlatBiases(flatBiases);

  newState->setDbmModel(model);
}

}

}
