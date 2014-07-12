/*
 * ReconstructionTest_gpu.cu
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#include "ReconstructionTest.h"

#include <tbblas/deeplearn/dbn.hpp>
#include <tbblas/rearrange.hpp>

#include <iostream>

namespace gml {

namespace dbn {

ReconstructionTestChecker::ReconstructionTestChecker() {
  ReconstructionTest test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Dataset, test);
  CHECK_MEMORY_LAYOUT2(Reconstructions, test);
  CHECK_MEMORY_LAYOUT2(MaxLayer, test);
}

void ReconstructionTest::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  typedef dbn_t::value_t value_t;
  const unsigned dimCount = dbn_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;


  v_host_tensor_t& dataset = *getDataset();
  boost::shared_ptr<v_host_tensor_t> reconstructions(new v_host_tensor_t());

  tbblas::deeplearn::dbn<value_t, dimCount> dbn(*getModel());
  tensor_t v;

  for (size_t i = 0; i < dataset.size(); ++i) {
    dim_t block = dataset[i]->size() / getModel()->crbms()[0]->visible_bias().size();
    block[dimCount - 1] = 1;
    v = *dataset[i];

    dbn.cvisibles() = rearrange(v, block);
    dbn.normalize_visibles();
    dbn.infer_hiddens(getMaxLayer());
    dbn.infer_visibles(getMaxLayer());
    dbn.diversify_visibles();

    v = rearrange_r(dbn.cvisibles(), block);
    reconstructions->push_back(boost::make_shared<host_tensor_t>(v));

    if (monitor)
      monitor->reportProgress((double)(i+1) / dataset.size());
  }

  newState->setReconstructions(reconstructions);
}

}

}
