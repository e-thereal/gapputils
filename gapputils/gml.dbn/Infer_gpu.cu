/*
 * Infer_gpu.cu
 *
 *  Created on: Oct 28, 2014
 *      Author: tombr
 */

#include "Infer.h"

#include <tbblas/deeplearn/dbn.hpp>
#include <tbblas/rearrange.hpp>

#include <tbblas/dot.hpp>
#include <tbblas/math.hpp>
#include <tbblas/util.hpp>
#include <tbblas/new_context.hpp>
#include <tbblas/change_stream.hpp>

#include <boost/thread/thread.hpp>

#include <omp.h>

namespace gml {

namespace dbn {

InferChecker::InferChecker() {
  Infer test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(InputTensors, test);
  CHECK_MEMORY_LAYOUT2(InputUnits, test);
  CHECK_MEMORY_LAYOUT2(Layer, test);
  CHECK_MEMORY_LAYOUT2(TopDown, test);
  CHECK_MEMORY_LAYOUT2(GpuCount, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchLength, test);
  CHECK_MEMORY_LAYOUT2(OutputTensors, test);
  CHECK_MEMORY_LAYOUT2(OutputUnits, test);
}

void Infer::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  using namespace tbblas;
  using namespace tbblas::deeplearn;

  typedef dbn_t::value_t value_t;
  const unsigned dimCount = dbn_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  int layer = getLayer();
  const int clayers = getModel()->crbms().size(), dlayers = getModel()->rbms().size(), layers = clayers + dlayers;
  if (layer < 0)
    layer = layers;

  if (layer > layers) {
    dlog(Severity::Warning) << "Invalid layer specified. The DBN has only " << layers << " layers. Aborting!";
    return;
  }

  if (!getTopDown()) {
    if (!getInputTensors()) {
      dlog(Severity::Warning) << "Input tensors are required for bottom-up inference. Aborting!";
      return;
    }

    v_host_tensor_t& dataset = *getInputTensors();

    boost::shared_ptr<v_host_tensor_t> tensors(new v_host_tensor_t(dataset.size()));
    boost::shared_ptr<v_data_t> hiddens(new v_data_t(dataset.size()));

    omp_set_num_threads(getGpuCount());
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

        dbn.set_input(v1);
        dbn.normalize_visibles();
        dbn.infer_hiddens(getLayer());

        cudaStreamSynchronize(copyStream);
        if (layer == 0)
          v2 = dbn.cvisibles();
        else if (layer <= clayers)
          v2 = dbn.chiddens(layer - 1);
        tbblas::synchronize();

        if (layer <= clayers){
          change_stream context(copyStream);
          tensors->at(i) = boost::make_shared<host_tensor_t>(v2);
        } else {
          boost::shared_ptr<data_t> hidden(new data_t(dbn.hiddens(layer - clayers - 1).count()));
          thrust::copy(dbn.hiddens(layer - clayers - 1).begin(), dbn.hiddens(layer - clayers - 1).end(), hidden->begin());
          hiddens->at(i) = hidden;
        }

        #pragma omp master
        if (monitor)
          monitor->reportProgress((double)(i+1) / (double)dataset.size() * 100.0);
      }
      cudaStreamSynchronize(copyStream);
      cudaStreamDestroy(copyStream);
    }

    if (layer <= clayers)
      newState->setOutputTensors(tensors);
    else
      newState->setOutputUnits(hiddens);
  } else {

    int count = 0;
    if (layer <= clayers) {
      if (!getInputTensors()) {
        dlog(Severity::Warning) << "InputTensors required to reconstruct from layer " << layer << ". Aborting!";
        return;
      }
      count = getInputTensors()->size();
    } else {
      if (!getInputUnits()) {
        dlog(Severity::Warning) << "InputUnits required to reconstruct from layer " << layer << ". Aborting!";
        return;
      }
      count = getInputUnits()->size();
    }

    boost::shared_ptr<v_host_tensor_t> tensors(new v_host_tensor_t(count));

    omp_set_num_threads(getGpuCount());

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

      if (layer > clayers) {
        dbn.hiddens(layer - clayers - 1).resize(seq(1, (int)getInputUnits()->at(0)->size()));
      }

      tensor_t h1, h2;
      tensor_t htemp;

      if (layer <= clayers && count > tid)
        htemp = *getInputTensors()->at(tid);

      for (size_t i = tid; i < count; i += getGpuCount()) {
        if (layer <= clayers) {
          h1 = htemp;
          tbblas::synchronize();

          if (i + getGpuCount() < count) {
            change_stream context(copyStream);
            htemp = *getInputTensors()->at(i + getGpuCount());
          }

          dbn.chiddens(layer - 1) = h1;
        } else {
          thrust::copy(getInputUnits()->at(i)->begin(), getInputUnits()->at(i)->end(), dbn.hiddens(layer - clayers - 1).begin());
        }

        dbn.infer_visibles();
        dbn.diversify_visibles();

        cudaStreamSynchronize(copyStream);
        dbn.get_input(h2);
        tbblas::synchronize();

        {
          change_stream context(copyStream);
          tensors->at(i) = boost::make_shared<host_tensor_t>(h2);
        }

        #pragma omp master
        if (monitor)
          monitor->reportProgress((double)(i+1) / (double)count * 100.0);
      }
      cudaStreamSynchronize(copyStream);
      cudaStreamDestroy(copyStream);
    }

    newState->setOutputTensors(tensors);
  }
}

}

}
