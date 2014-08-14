/*
 * FineTuning_gpu.cu
 *
 *  Created on: Jul 21, 2014
 *      Author: tombr
 */

#include "FineTuning.h"

#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/rearrange.hpp>
#include <tbblas/deeplearn/dbn.hpp>
#include <tbblas/new_context.hpp>
#include <tbblas/util.hpp>

#include <omp.h>

namespace gml {

namespace dbn {

FineTuningChecker::FineTuningChecker() {
  FineTuning test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(Tensors, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(GpuCount, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchLength, test);

  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(InitialMomentum, test);
  CHECK_MEMORY_LAYOUT2(FinalMomentum, test);
  CHECK_MEMORY_LAYOUT2(MomentumDecayEpochs, test);
  CHECK_MEMORY_LAYOUT2(WeightDecay, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);

  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(ReconstructionError, test);
}

void FineTuning::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef dbn_t::value_t value_t;
  const unsigned dimCount = dbn_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  const size_t batchSize = getBatchSize();
  const size_t batchCount = getTensors()->size() / batchSize;
  const size_t epochCount = getEpochCount();

  boost::shared_ptr<dbn_t> model(new dbn_t(*getInitialModel()));
  if (model->crbms().size() == 0) {
    dlog(Severity::Warning) << "At least one convolutional layer required. Aborting!";
    return;
  }

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (getGpuCount() > deviceCount) {
    dlog(Severity::Warning) << "Only " << deviceCount << " CUDA-enabled devices found, where " << getGpuCount() << " are required according to GpuCount. Aborting!";
    return;
  }

  // Initialize constants
  value_t epsilonw =  getLearningRate() / batchSize;  // Learning rate for weights
  value_t epsilonvb = getLearningRate() / batchSize;  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate() / batchSize;  // Learning rate for biases of hidden units
  value_t weightcost = getWeightDecay() * getLearningRate();
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;
  size_t voxelCount = sum(model->crbms()[0]->mask()) * model->crbms()[0]->visibles_size()[dimCount - 1];

  std::vector<boost::shared_ptr<host_tensor_t> >& X = *getTensors();
  dim_t block = X[0]->size() / model->crbms()[0]->visible_bias().size();
  block[dimCount - 1] = 1;

  omp_set_dynamic(false);
  omp_set_num_threads(getGpuCount());

  tbblas::enable_peer_access(getGpuCount());

  value_t totalError = 0;

  std::vector<tbblas::deeplearn::dbn<value_t, dimCount>* > dbns(getGpuCount());

  #pragma omp parallel
  {
    const size_t tid = omp_get_thread_num();
    cudaSetDevice(tid);
    new_context context;

    dbn_t tempModel(*model);
    tbblas::deeplearn::dbn<value_t, dimCount> dbn(tid == 0 ? *model : tempModel);
    for (size_t i = 0; i < model->crbms().size() && i < getFilterBatchLength().size(); ++i)
      dbn.set_batch_length(i, getFilterBatchLength()[i]);

    dbns[tid] = &dbn;

    value_t error = 0;
    tensor_t v, vtemp;

    dbn.allocate_gpu_memory();
    tbblas::synchronize();
    #pragma omp barrier

    #pragma omp master
    dlog() << "Trainer initialized. Starting training.";

    for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
      error = 0;

      #pragma omp barrier

      #pragma omp master
      {
        totalError = 0;

        if (iEpoch < getMomentumDecayEpochs()) {
          const value_t t = (value_t)iEpoch / (value_t)getMomentumDecayEpochs();
          momentum = (1 - t) * initialmomentum + t * finalmomentum;
        } else {
          momentum = finalmomentum;
        }
      }
      #pragma omp barrier

      for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {

        // Apply momentum for next batch
        dbn.init_gradient_updates(momentum, weightcost);

        for (size_t iSample = tid; iSample < batchSize; iSample += getGpuCount()) {
  //        crbm.init_dropout(getHiddenDropout(), getDropoutMethod());

          // Get new sample
          if (getRandomizeTraining())
            vtemp = *X[rand() % X.size()];
          else
            vtemp = *X[iSample + iBatch * batchSize];
          dbn.cvisibles() = rearrange(vtemp, block);
          dbn.normalize_visibles();

          v = dbn.cvisibles();

          dbn.infer_hiddens();
          dbn.update_positive_gradient(epsilonw, epsilonvb, epsilonhb);

          dbn.sample_hiddens();
          dbn.sample_visibles();
          dbn.infer_visibles();
          dbn.infer_hiddens();
          dbn.update_negative_gradient(epsilonw, epsilonvb, epsilonhb);
          error += sqrt(dot((dbn.cvisibles() - v), (dbn.cvisibles() - v)) / voxelCount);
        } /* end of sample */

        // Accumulate model increments
        tbblas::synchronize();
        #pragma omp barrier

        if (tid == 0 && getGpuCount() == 2)
          dbn.accumulate_gradients(*dbns[1]);

        tbblas::synchronize();
        #pragma omp barrier

        dbn.apply_gradient();

        if (monitor)
          monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
      } /* end of batch */

      tbblas::synchronize();
      #pragma omp barrier

      #pragma omp critical
      totalError += error;

      #pragma omp master
      {
        dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << totalError / X.size();
        if (monitor)
          monitor->reportProgress(100. * (iEpoch + 1) / epochCount);
      }
    } /* end of epochs */
  } /* end of parallel */

  disable_peer_access(getGpuCount());

//    newState->setAverageEpochTime(_timer.elapsed() / getEpochCount());
  newState->setReconstructionError(totalError / X.size());
  newState->setModel(model);
}

}

}
