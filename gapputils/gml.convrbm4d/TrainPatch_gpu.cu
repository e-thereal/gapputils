/*
 * TrainPatch_gpu.cu
 *
 *  Created on: Oct 16, 2014
 *      Author: tombr
 */

//#define TBBLAS_INTERRUPT_ALLOC_ENABLED
//#define TBBLAS_ALLOC_WARNING_ENABLED

#include "TrainPatch.h"

#include <tbblas/math.hpp>
#include <tbblas/random.hpp>
#include <tbblas/zeros.hpp>
#include <tbblas/ones.hpp>
#include <tbblas/serialize.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/repeat.hpp>
#include <tbblas/io.hpp>
#include <tbblas/rearrange.hpp>

#include <boost/timer.hpp>

#include <fstream>

#include <tbblas/deeplearn/conv_rbm.hpp>

namespace gml {

namespace convrbm4d {

TrainPatchChecker::TrainPatchChecker() {
  TrainPatch trainer;
  trainer.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(DbmLayer, trainer);

  CHECK_MEMORY_LAYOUT2(PatchCount, trainer);
  CHECK_MEMORY_LAYOUT2(EpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(BatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(GpuCount, trainer);

  CHECK_MEMORY_LAYOUT2(SparsityMethod, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);

  CHECK_MEMORY_LAYOUT2(CdIterations, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRate, trainer);
  CHECK_MEMORY_LAYOUT2(LearningDecay, trainer);
  CHECK_MEMORY_LAYOUT2(InitialMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(FinalMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(MomentumDecayEpochs, trainer);
  CHECK_MEMORY_LAYOUT2(WeightDecay, trainer);
  CHECK_MEMORY_LAYOUT2(WeightVectorLimit, trainer);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, trainer);
  CHECK_MEMORY_LAYOUT2(DropoutMethod, trainer);
  CHECK_MEMORY_LAYOUT2(VisibleDropout, trainer);
  CHECK_MEMORY_LAYOUT2(HiddenDropout, trainer);
  CHECK_MEMORY_LAYOUT2(FilterDropout, trainer);
  CHECK_MEMORY_LAYOUT2(CalculateError, trainer);
  CHECK_MEMORY_LAYOUT2(UpdateModel, trainer);

  CHECK_MEMORY_LAYOUT2(CurrentEpoch, trainer);
  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(AverageEpochTime, trainer);
  CHECK_MEMORY_LAYOUT2(ReconstructionError, trainer);
  CHECK_MEMORY_LAYOUT2(Patches, trainer);
}

#define START size_t timerCycles = getEpochCount(); \
    boost::timer _timer;

#define STOP { \
    cudaStreamSynchronize(0); \
    std::cout << __LINE__ << ": " << _timer.elapsed() << std::endl; \
    _timer.restart(); \
}

#define TIMER_LOOP for(size_t iCycle = 0; iCycle < timerCycles; ++iCycle)

#define TRACE std::cout << __LINE__ << std::endl;

#define DROPOUT

void TrainPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef float value_t;
  const unsigned dimCount = model_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  /*** SETUP TRAINING PARAMETERS ***/

  std::vector<boost::shared_ptr<host_tensor_t> >& X = *getTensors();

  const size_t batchSize = getBatchSize();
  const size_t filterBatchLength = getFilterBatchSize();
  const size_t batchCount = getTensors()->size() / batchSize;
  const size_t epochCount = getEpochCount();

  if (filterBatchLength > getInitialModel()->filters().size() ||
      getInitialModel()->filters().size() % filterBatchLength != 0)
  {
    dlog(Severity::Warning) << "Invalid FilterBatchSize. Aborting!";
    return;
  }

  /*** PREPARE MASTER THREAD ***/

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));
  model->set_shared_bias(true);

  dim_t patchSize = model->input_size();
  dim_t range = X[0]->size() - patchSize + 1;

  tbblas::deeplearn::conv_rbm<float, 4> crbm(*model, getGpuCount());
  crbm.set_batch_length(getFilterBatchSize());
  crbm.set_sparsity_method(getSparsityMethod());
  crbm.set_sparsity_target(getSparsityTarget());
  crbm.set_sparsity_weight(getSparsityWeight());

  // Prepare sizes
  size_t voxelCount = model->visible_bias().count();

  // Initialize constants
  value_t epsilonw =  getLearningRate() / batchSize / getPatchCount();  // Learning rate for weights
  value_t epsilonvb = getLearningRate() / batchSize / getPatchCount();  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate() / batchSize / getPatchCount();  // Learning rate for biases of hidden units
  value_t weightcost = getWeightDecay() * getLearningRate();
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;

  value_t error = 0;
  tensor_t v, sample;

  /*** START OF PARALLEL CODE ***/

  crbm.allocate_gpu_memory();

  dlog() << "Trainer initialized. Starting training.";
  boost::shared_ptr<v_host_tensor_t> patches(new v_host_tensor_t());
  newState->setPatches(patches);

  for (size_t iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
    error = 0;

    if (iEpoch < getMomentumDecayEpochs()) {
      const value_t t = (value_t)iEpoch / (value_t)getMomentumDecayEpochs();
      momentum = (1 - t) * initialmomentum + t * finalmomentum;
    } else {
      momentum = finalmomentum;
    }

    for (size_t iBatch = 0; iBatch < batchCount; ++iBatch) {

      // Apply momentum for next batch
      crbm.init_gradient_updates(momentum, weightcost);

      for (size_t iSample = 0; iSample < batchSize; ++iSample) {

        if (getRandomizeTraining())
          sample = *X[rand() % X.size()];
        else
          sample = *X[iSample + iBatch * batchSize];

        for (size_t iPatch = 0; iPatch < getPatchCount(); ++iPatch) {
          crbm.init_dropout(getHiddenDropout(), getDropoutMethod());

          // Get new patch
          dim_t topleft = seq(rand() % range[0], rand() % range[1], rand() % range[2], 0);
          crbm.visibles() = rearrange(sample[topleft, patchSize], model->stride_size());
          crbm.normalize_visibles();

          if (getCalculateError()) {
            v = crbm.visibles();
          }

          /*** BEGIN OF POSITIVE PHASE ***/

          crbm.infer_hiddens();
          crbm.update_positive_gradient(epsilonw, epsilonvb, epsilonhb);

          for (size_t iCd = 0; iCd < getCdIterations(); ++iCd) {
            crbm.sample_hiddens();
            crbm.sample_visibles();
          } /* end of cd iterations */

          /*** RECONSTRUCT FROM SAMPLES ***/

          crbm.infer_hiddens();
          crbm.update_negative_gradient(epsilonw, epsilonvb, epsilonhb);

          if (getCalculateError()) {
            error += sqrt(dot((crbm.visibles() - v), (crbm.visibles() - v)) / voxelCount);
          }

          if (iEpoch == epochCount - 1 && iBatch == 0)
            patches->push_back(boost::make_shared<host_tensor_t>(crbm.visibles()));

        } /* end of patches */
      } /* end of sample */

      crbm.apply_gradient();

      if (monitor)
        monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
    } /* end of batch */

    epsilonw *= getLearningDecay();
    epsilonvb *= getLearningDecay();
    epsilonhb *= getLearningDecay();

    if (getCalculateError())
      dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / X.size() / getPatchCount();
    else
      dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / epochCount, getUpdateModel() && (iEpoch % getUpdateModel() == 0));
  } /* end of epochs */

//    newState->setAverageEpochTime(_timer.elapsed() / getEpochCount());
  newState->setReconstructionError(error / X.size() / getPatchCount());
  newState->setModel(model);
}

}

}
