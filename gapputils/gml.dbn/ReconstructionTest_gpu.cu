/*
 * ReconstructionTest_gpu.cu
 *
 *  Created on: Jul 11, 2014
 *      Author: tombr
 */

#include "ReconstructionTest.h"

#include <tbblas/deeplearn/dbn.hpp>
#include <tbblas/rearrange.hpp>

#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>
#include <tbblas/util.hpp>
#include <tbblas/new_context.hpp>
#include <tbblas/change_stream.hpp>

#include <boost/thread/thread.hpp>

#include <omp.h>

#include <iostream>

namespace gml {

namespace dbn {

ReconstructionTestChecker::ReconstructionTestChecker() {
  ReconstructionTest test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Dataset, test);
  CHECK_MEMORY_LAYOUT2(Type, test);
  CHECK_MEMORY_LAYOUT2(MaxLayer, test);
  CHECK_MEMORY_LAYOUT2(GpuCount, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchLength, test);
  CHECK_MEMORY_LAYOUT2(Reconstructions, test);
  CHECK_MEMORY_LAYOUT2(ReconstructionError, test);
}

void load_v(const tbblas::tensor<dbn_t::value_t, dbn_t::dimCount>* from,
    tbblas::tensor<dbn_t::value_t, dbn_t::dimCount, true>* to, cudaStream_t stream)
{
  tbblas::change_stream context(stream);
  *to = *from;
}

void ReconstructionTest::update(IProgressMonitor* monitor) const {
  using namespace tbblas;
  using namespace tbblas::deeplearn;

  typedef dbn_t::value_t value_t;
  const unsigned dimCount = dbn_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  v_host_tensor_t& dataset = *getDataset();
  boost::shared_ptr<v_host_tensor_t> reconstructions(new v_host_tensor_t(dataset.size()));

  dim_t block = dataset[0]->size() / getModel()->crbms()[0]->visible_bias().size();
  block[dimCount - 1] = 1;

  omp_set_num_threads(getGpuCount());

  value_t totalError = 0;

  #pragma omp parallel
  {
    size_t tid = omp_get_thread_num();

    cudaSetDevice(tid);

    new_context context;
    cudaStream_t copyStream;
    cudaStreamCreate(&copyStream);
    tbblas::deeplearn::dbn<value_t, dimCount> dbn(*getModel());

    for (size_t i = 0; i < getModel()->crbms().size() && i < getFilterBatchLength().size(); ++i)
      dbn.set_batch_length(i, getFilterBatchLength()[i]);

    tensor_t v1, v2;
    value_t error = 0;

    tensor_t vtemp;

    if (dataset.size() > tid)
      vtemp = *dataset[tid];

    for (size_t i = tid; i < dataset.size(); i += getGpuCount()) {
      v1 = vtemp;
      tbblas::synchronize();

      if (i + getGpuCount() < dataset.size()) {
        change_stream context(copyStream);
        vtemp = *dataset[i + getGpuCount()];
      }

      dbn.cvisibles() = rearrange(v1, block);
      dbn.normalize_visibles();
      dbn.infer_hiddens(getMaxLayer());
      dbn.infer_visibles(getMaxLayer());
      dbn.diversify_visibles();

      cudaStreamSynchronize(copyStream);
      v2 = rearrange_r(dbn.cvisibles(), block);
      tbblas::synchronize();

      switch (getType()) {
      case TestType::Reconstruct:
        {
          change_stream context(copyStream);
          reconstructions->at(i) = boost::make_shared<host_tensor_t>(v2);
        }
        break;

      case TestType::CalculateMSE:
        error += dot(v1 - v2, v1 - v2) / v1.count();
        break;

      case TestType::CalculateRMSE:
        error += sqrt(dot(v1 - v2, v1 - v2) / v1.count());
        break;

      case TestType::CalculateRRMSE:
        error += sqrt(dot(v1 - v2, v1 - v2) / v1.count()) / (sum(v1) / v1.count());
        break;
      }

      #pragma omp master
      if (monitor)
        monitor->reportProgress((double)(i+1) / (double)dataset.size() * 100.0);
    }
    cudaStreamSynchronize(copyStream);

    #pragma omp critical
    totalError += error;
    #pragma omp barrier

    #pragma omp master
    switch (getType()) {
    case TestType::Reconstruct:
      newState->setReconstructions(reconstructions);
      break;

    case TestType::CalculateMSE:
    case TestType::CalculateRMSE:
    case TestType::CalculateRRMSE:
      newState->setReconstructionError(totalError / dataset.size());
      break;
    }

    cudaStreamDestroy(copyStream);
  }
}

}

}
