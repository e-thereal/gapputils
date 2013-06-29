/*
 * CreateDbm_gpu.cu
 *
 *  Created on: Jun 28, 2013
 *      Author: tombr
 */

#include "CreateDbm.h"

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
  boost::shared_ptr<DbmModel> model(new DbmModel());

  // Initialize DBM from several RBMs
  // - Copy filters
  // - Average biases of intermediate layers
  // - Copy masks

  boost::shared_ptr<vv_host_tensor_t> weights(new vv_host_tensor_t());
  boost::shared_ptr<v_host_tensor_t> masks(new v_host_tensor_t());

  std::vector<boost::shared_ptr<Model> >& crbms = *getCrbmModels();
  for (size_t iLayer = 0; iLayer < crbms.size(); ++iLayer) {
    v_host_tensor_t& oldFilters = *crbms[iLayer]->getFilters();

    boost::shared_ptr<v_host_tensor_t> newFilters(new v_host_tensor_t());
    for (size_t iFilter = 0; oldFilters.size(); ++iFilter) {
      newFilters->push_back(boost::make_shared<host_tensor_t>(*oldFilters[iFilter]));
    }
    weights->push_back(newFilters);
    masks->push_back(boost::make_shared<host_tensor_t>(*crbms[iLayer]->getMask()));
  }
  model->setWeights(weights);
  model->setMasks(masks);

  newState->setDbmModel(model);
}

}

}
