/*
 * Trainer_gpu.cu
 *
 *  Created on: Nov 22, 2012
 *      Author: tombr
 */

//#define TBBLAS_INTERRUPT_ALLOC_ENABLED
//#define TBBLAS_ALLOC_WARNING_ENABLED

#include "Trainer.h"

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

TrainerChecker::TrainerChecker() {
  Trainer trainer;
  trainer.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(DbmLayer, trainer);
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
  CHECK_MEMORY_LAYOUT2(ShareBiasTerms, trainer);
  CHECK_MEMORY_LAYOUT2(DropoutMethod, trainer);
  CHECK_MEMORY_LAYOUT2(VisibleDropout, trainer);
  CHECK_MEMORY_LAYOUT2(HiddenDropout, trainer);
  CHECK_MEMORY_LAYOUT2(FilterDropout, trainer);
  CHECK_MEMORY_LAYOUT2(CalculateError, trainer);
  CHECK_MEMORY_LAYOUT2(UpdateModel, trainer);

  CHECK_MEMORY_LAYOUT2(FindLearningRate, trainer);
  CHECK_MEMORY_LAYOUT2(TrialLearningRates, trainer);
  CHECK_MEMORY_LAYOUT2(TrialEpochCount, trainer);

  CHECK_MEMORY_LAYOUT2(CurrentEpoch, trainer);
  CHECK_MEMORY_LAYOUT2(Model, trainer);
  CHECK_MEMORY_LAYOUT2(AverageEpochTime, trainer);
  CHECK_MEMORY_LAYOUT2(ReconstructionError, trainer);
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

void Trainer::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  typedef float value_t;
  const unsigned dimCount = model_t::dimCount;
  typedef tensor<value_t, dimCount, true> tensor_t;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  /*** SETUP TRAINING PARAMETERS ***/

  const size_t batchSize = getBatchSize();
  const size_t filterBatchLength = getFilterBatchSize();
  const size_t batchCount = getTensors()->size() / batchSize;

  size_t epochCount = getEpochCount();

  std::vector<boost::shared_ptr<host_tensor_t> >& X = *getTensors();

  if (filterBatchLength > getInitialModel()->filters().size() ||
      getInitialModel()->filters().size() % filterBatchLength != 0)
  {
    dlog(Severity::Warning) << "Invalid FilterBatchSize. Aborting!";
    return;
  }

  if (getFindLearningRate() && !getCalculateError()) {
    dlog(Severity::Warning) << "Error calculation must be turned on in order to find the best learning rate. Aborting!";
    return;
  }

  /*** PREPARE MASTER THREAD ***/

  // Initialize constants
  value_t weightcost = getWeightDecay() * getLearningRate();
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;

  value_t epsilonw =  getLearningRate() / batchSize;  // Learning rate for weights
  value_t epsilonvb = getBiasLearningRate() / batchSize;  // Learning rate for biases of visible units
  value_t epsilonhb = getBiasLearningRate() / batchSize;  // Learning rate for biases of hidden units

  std::vector<double> learningRates = getTrialLearningRates();
  value_t bestEpsilon, bestError;

  if (!getFindLearningRate())
    learningRates.clear();

  for (int iLearningRate = 0; iLearningRate < learningRates.size() + 1; ++iLearningRate) {
    if (iLearningRate < learningRates.size()) {
      dlog(Severity::Message) << "Trying learning rate of " << learningRates[iLearningRate];
      epsilonhb = epsilonvb = epsilonw = learningRates[iLearningRate] / batchSize;
      epochCount = getTrialEpochCount();
    } else {
      if (learningRates.size()) {
        epsilonhb = epsilonvb = epsilonw = bestEpsilon;
      } else {
        epsilonw =  getLearningRate() / batchSize;
        epsilonvb = getBiasLearningRate() / batchSize;
        epsilonhb = getBiasLearningRate() / batchSize;
      }
      dlog(Severity::Message) << "Final run with learning rate: " << bestEpsilon * batchSize;
      epochCount = getEpochCount();
    }

    boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));
    model->set_shared_bias(getShareBiasTerms());

    tbblas::deeplearn::conv_rbm_trainer<float, 4> crbm(*model, getGpuCount());
    crbm.set_batch_length(getFilterBatchSize());
    crbm.set_sparsity_method(getSparsityMethod());
    crbm.set_sparsity_target(getSparsityTarget());
    crbm.set_sparsity_weight(getSparsityWeight());

    // Prepare sizes
    size_t voxelCount = sum(model->mask()) * model->visibles_size()[dimCount - 1];

    value_t error = 0, learningDecay = 1;
    tensor_t v, input;

    /*** START OF PARALLEL CODE ***/

    crbm.allocate_gpu_memory();

    dlog() << "Trainer initialized. Starting training.";

    for (size_t iEpoch = 0; iEpoch < epochCount && error == error && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {
      error = 0;

      // Learning decay only during final learning
      if (getLearningDecay() > 1 && iLearningRate == learningRates.size()) {
        learningDecay = (value_t)getLearningDecay() / ((value_t)getLearningDecay() + (value_t)iEpoch);
      }

      if (iEpoch < getMomentumDecayEpochs()) {
        const value_t t = (value_t)iEpoch / (value_t)getMomentumDecayEpochs();
        momentum = (1 - t) * initialmomentum + t * finalmomentum;
      } else {
        momentum = finalmomentum;
      }

      for (size_t iBatch = 0; iBatch < batchCount && error == error; ++iBatch) {

        // Apply momentum for next batch
        crbm.init_gradient_updates(momentum, weightcost);

        for (size_t iSample = 0; iSample < batchSize && error == error; ++iSample) {
          crbm.init_dropout(getHiddenDropout(), getDropoutMethod());

          // Get new sample
          if (getRandomizeTraining())
            input = *X[rand() % X.size()];
          else
            input = *X[iSample + iBatch * batchSize];

          crbm.visibles() = rearrange(input, model->stride_size());
          crbm.normalize_visibles();

          if (getCalculateError())
            v = crbm.visibles();

          /*** BEGIN OF POSITIVE PHASE ***/
          crbm.infer_hiddens();
          crbm.update_positive_gradient(epsilonw * learningDecay, epsilonvb * learningDecay, epsilonhb * learningDecay);

          for (size_t iCd = 0; iCd < getCdIterations(); ++iCd) {
            crbm.sample_hiddens();
            crbm.sample_visibles();
          } /* end of cd iterations */

          /*** RECONSTRUCT FROM SAMPLES ***/

          crbm.infer_hiddens();
          crbm.update_negative_gradient(epsilonw * learningDecay, epsilonvb * learningDecay, epsilonhb * learningDecay);

          if (getCalculateError()) {
            error += sqrt(dot((crbm.visibles() - v), (crbm.visibles() - v)) / voxelCount);
          }
        } /* end of sample */

        crbm.apply_gradient();

        if (monitor)
          monitor->reportProgress(100. * (iEpoch * batchCount + (iBatch + 1)) / (epochCount * batchCount));
      } /* end of batch */

      if (getCalculateError())
        dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / X.size();
      else
        dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

      if (monitor)
        monitor->reportProgress(100. * (iEpoch + 1) / epochCount, getUpdateModel() && (iEpoch % getUpdateModel() == 0));
    } /* end of epochs */

  //    newState->setAverageEpochTime(_timer.elapsed() / epochCount);

    if (iLearningRate < learningRates.size()) {
      if (iLearningRate == 0 || !(error > bestError)) {   // using not greater instead of lesser to handle nan case.
        bestError = error;
        bestEpsilon = epsilonw;
        dlog(Severity::Message) << "Found better learning rate: " << epsilonw * batchSize << " with an error of " << bestError / X.size() << ".";
      }
    } else {
      newState->setReconstructionError(error / X.size());
      newState->setModel(model);
    }
  }
}

}

}
