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

#include <tbblas/deeplearn/conv_rbm_trainer.hpp>

namespace gml {

namespace convrbm4d {

TrainPatchChecker::TrainPatchChecker() {
  TrainPatch trainer;
  trainer.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(DbmLayer, trainer);

  CHECK_MEMORY_LAYOUT2(SuperPatchWidth, trainer);
  CHECK_MEMORY_LAYOUT2(SuperPatchHeight, trainer);
  CHECK_MEMORY_LAYOUT2(SuperPatchDepth, trainer);
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

  /*** PREPARE SUPER PATCH TRAINING ***/

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));
  model->set_shared_bias(true);

  // TODO: handle training with differently sized images

  if (X[0]->size()[3] != model->input_size()[3]) {
    dlog(Severity::Warning) << "Number of channels doesn't match. Aborting!";
    return;
  }

  // Change size of bias terms to the super patch size and unstride the model
  dim_t superPatchSize = seq(
      getSuperPatchWidth() > 0 ? getSuperPatchWidth() : X[0]->size()[0],
      getSuperPatchHeight() > 0 ? getSuperPatchHeight() : X[0]->size()[1],
      getSuperPatchDepth() > 0 ? getSuperPatchDepth() : X[0]->size()[2],
      X[0]->size()[3]);
  dim_t superPatchLayerSize = superPatchSize;
  superPatchLayerSize[dimCount - 1] = 1;

  dim_t patchSize = model->input_size();
  dim_t oldStride = model->stride_size();

  model->change_stride(seq<dimCount>(1));
  model->change_size(superPatchSize);

  dim_t superPatchStepSize = model->hiddens_size();
  dim_t superPatchMaxStep = X[0]->size();
  if (model->convolution_type() == deeplearn::convolution_type::Valid) {
    superPatchMaxStep = superPatchMaxStep - model->kernel_size() + 1;
  }
  superPatchStepSize[dimCount - 1] = 1;

  tbblas::deeplearn::conv_rbm_trainer<float, 4> crbm(*model, getGpuCount());
  crbm.set_batch_length(getFilterBatchSize());
  crbm.set_sparsity_method(getSparsityMethod());
  crbm.set_sparsity_target(getSparsityTarget());
  crbm.set_sparsity_weight(getSparsityWeight());

  // Prepare sizes
  size_t voxelCount = model->visible_bias().count();

  // Initialize constants
  value_t epsilonw =  getLearningRate() / batchSize / superPatchMaxStep.prod() * superPatchStepSize.prod();// / getPatchCount();  // Learning rate for weights
  value_t epsilonvb = getLearningRate() / batchSize / superPatchMaxStep.prod() * superPatchStepSize.prod();// / getPatchCount();  // Learning rate for biases of visible units
  value_t epsilonhb = getLearningRate() / batchSize / superPatchMaxStep.prod() * superPatchStepSize.prod();// / getPatchCount();  // Learning rate for biases of hidden units
  value_t weightcost = getWeightDecay() * getLearningRate();
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;

  value_t error = 0;
  tensor_t v, sample;
  host_tensor_t overlapMask;

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

    for (size_t iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

      // Apply momentum for next batch
      crbm.init_gradient_updates(momentum, weightcost);

      for (size_t iSample = 0; iSample < batchSize; ++iSample) {

        if (getRandomizeTraining())
          sample = *X[rand() % X.size()];
        else
          sample = *X[iSample + iBatch * batchSize];

        for (int z = 0; z < superPatchMaxStep[2]; z += superPatchStepSize[2]) {
          for (int y = 0; y < superPatchMaxStep[1]; y += superPatchStepSize[1]) {
            for (int x = 0; x < superPatchMaxStep[0]; x += superPatchStepSize[0]) {

              crbm.init_dropout(getHiddenDropout(), getDropoutMethod());

              // Get new patch
              dim_t topleft = seq(x, y, z, 0);
              dim_t overlap = min(superPatchSize, sample.size() - topleft);   // the overlap of the current super patch and the image
              for (size_t i = 0; i < dimCount; ++i)
                assert(overlap[i] > 0);

              dim_t overlapMaskSize = overlap;
              overlapMaskSize[dimCount - 1] = 1;

              overlapMask = zeros<value_t>(superPatchLayerSize);
              overlapMask[seq<dimCount>(0), overlapMaskSize] = ones<value_t>(overlapMaskSize);
              crbm.change_mask(overlapMask);

              crbm.visibles() = zeros<value_t>(superPatchSize);
              crbm.visibles()[seq<dimCount>(0), overlap] = sample[topleft, overlap];

              if (iEpoch == epochCount - 1 && iBatch == 0)
                patches->push_back(boost::make_shared<host_tensor_t>(crbm.visibles()));

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
            }
          }
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
      dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / X.size() / superPatchMaxStep.prod() * superPatchStepSize.prod();
    else
      dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / epochCount, getUpdateModel() && (iEpoch % getUpdateModel() == 0));
  } /* end of epochs */

  crbm.write_model_to_host();

  // Change size of bias terms back to the patch size and stride the model
  model->change_size(patchSize);
  model->change_stride(oldStride);

//    newState->setAverageEpochTime(_timer.elapsed() / getEpochCount());
  newState->setReconstructionError(error / X.size() / superPatchMaxStep.prod() * superPatchStepSize.prod());
  newState->setModel(model);
}

}

}
