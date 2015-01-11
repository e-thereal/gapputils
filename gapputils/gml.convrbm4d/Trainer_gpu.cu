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
#include <tbblas/random.hpp>

#include <boost/timer.hpp>

#include <fstream>

#include <tbblas/deeplearn/conv_rbm.hpp>

namespace gml {

namespace convrbm4d {

TrainerChecker::TrainerChecker() {
  Trainer trainer;
  trainer.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, trainer);
  CHECK_MEMORY_LAYOUT2(Tensors, trainer);
  CHECK_MEMORY_LAYOUT2(DbmLayer, trainer);
  CHECK_MEMORY_LAYOUT2(EpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(TrialEpochCount, trainer);
  CHECK_MEMORY_LAYOUT2(BatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, trainer);
  CHECK_MEMORY_LAYOUT2(SubRegionCount, trainer);
  CHECK_MEMORY_LAYOUT2(GpuCount, trainer);

  CHECK_MEMORY_LAYOUT2(SparsityMethod, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityTarget, trainer);
  CHECK_MEMORY_LAYOUT2(SparsityWeight, trainer);

  CHECK_MEMORY_LAYOUT2(CdIterations, trainer);
  CHECK_MEMORY_LAYOUT2(Method, trainer);
  CHECK_MEMORY_LAYOUT2(LearningRates, trainer);
  CHECK_MEMORY_LAYOUT2(LearningDecay, trainer);
  CHECK_MEMORY_LAYOUT2(InitialMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(FinalMomentum, trainer);
  CHECK_MEMORY_LAYOUT2(MomentumDecayEpochs, trainer);
  CHECK_MEMORY_LAYOUT2(WeightDecay, trainer);
  CHECK_MEMORY_LAYOUT2(InitialWeights, trainer);
  CHECK_MEMORY_LAYOUT2(SignalToNoiseRatio, trainer);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, trainer);
  CHECK_MEMORY_LAYOUT2(ShareBiasTerms, trainer);
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

  if (getLearningRates().size() > 1 && !getCalculateError()) {
    dlog(Severity::Warning) << "Error calculation must be turned on in order to find the best learning rate. Aborting!";
    return;
  }

  /*** PREPARE MASTER THREAD ***/

  // Initialize constants
  value_t initialmomentum = getInitialMomentum();
  value_t finalmomentum = getFinalMomentum();
  value_t momentum;

  value_t epsilonw, initialWeight = 0;  // Learning rate for weights

  std::vector<double> learningRates = getLearningRates();
  std::vector<double> initialWeights = getInitialWeights();
  value_t bestEpsilon, bestError, bestWeight = 0;

  for (int iWeight = 0; iWeight < initialWeights.size() || (iWeight == 0 && initialWeights.size() == 0); ++iWeight) {
    for (int iLearningRate = 0; iLearningRate < learningRates.size() + 1; ++iLearningRate) {

      // iLearningRate == learningRate.size() marks the final run, this is only done when all the weights were tried
      if (iLearningRate == learningRates.size() && iWeight < (int)initialWeights.size() - 1) {
        continue;
      }

      // if only one weight and one learning rate is given, do the final run immediately
      if (iWeight == 0 && iLearningRate == 0 && initialWeights.size() <= 1 && learningRates.size() == 1 && _SignalToNoiseRatio <= 0) {
        bestEpsilon = learningRates[0];
        if (initialWeights.size())
          bestWeight = initialWeights[0];
        continue;
      }

      if (iLearningRate < learningRates.size()) {
        epsilonw = learningRates[iLearningRate];
        if (initialWeights.size())
          initialWeight = initialWeights[iWeight];
        epochCount = getTrialEpochCount();
        dlog(Severity::Message) << "Trying learning rate of " << learningRates[iLearningRate] << " and initial weight of " << initialWeight;
      } else {
        epsilonw = bestEpsilon;
        initialWeight = bestWeight;
        dlog(Severity::Message) << "Final run with learning rate: " << bestEpsilon << " and initial weight of " << initialWeight;
        epochCount = getEpochCount();
      }
      value_t weightcost = getWeightDecay();

      boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));
      model->set_shared_bias(getShareBiasTerms());

      // reset filters if initial weights are larger than 0
      if (initialWeight > 0) {
        random_tensor2<value_t, model_t::dimCount, false, normal<value_t> > randn(model->kernel_size());

        for (int iFilter = 0; iFilter < model->filters().size(); ++iFilter) {
          *model->filters()[iFilter] = initialWeight * randn();
        }
      }

      tbblas::deeplearn::conv_rbm<float, 4> crbm(*model, getSubRegionCount());
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
//          crbm.init_gradient_updates(momentum, weightcost);

          for (size_t iSample = 0; iSample < batchSize && error == error; ++iSample) {
            crbm.init_dropout(getHiddenDropout(), getDropoutMethod());

            // Get new sample
            if (getRandomizeTraining())
              input = *X[rand() % X.size()];
            else
              input = *X[iSample + iBatch * batchSize];

            crbm.visibles() = input;
            crbm.normalize_visibles();

            if (getCalculateError())
              v = crbm.visibles();

            /*** BEGIN OF POSITIVE PHASE ***/
            crbm.infer_hiddens();
            crbm.update_positive_gradient();

            for (size_t iCd = 0; iCd < getCdIterations(); ++iCd) {
              crbm.sample_hiddens();
              crbm.sample_visibles();
            } /* end of cd iterations */

            /*** RECONSTRUCT FROM SAMPLES ***/

            crbm.infer_hiddens();
            crbm.update_negative_gradient();

            if (getCalculateError()) {
              error += sqrt(dot((crbm.visibles() - v), (crbm.visibles() - v)) / voxelCount);
            }
          } /* end of sample */

          switch (getMethod()) {
          case TrainingMethod::Momentum:
            crbm.momentum_step(epsilonw * learningDecay, momentum, weightcost);
            break;

          case TrainingMethod::AdaDelta:
            crbm.adadelta_step(epsilonw * learningDecay, momentum, weightcost);
            break;
          }

          if (monitor) {
            const int totalEpochs = getTrialEpochCount() * max(1, (int)initialWeights.size()) * learningRates.size() + getEpochCount();
            const int currentEpoch = iEpoch + (iLearningRate + iWeight * learningRates.size()) * getTrialEpochCount();
            monitor->reportProgress(100. * (currentEpoch * batchCount + (iBatch + 1)) / (totalEpochs * batchCount));
          }
        } /* end of batch */

        if (getCalculateError())
          dlog(Severity::Trace) << "Error at epoch " << iEpoch << " of " << epochCount << ": " << error / X.size();
        else
          dlog(Severity::Trace) << "Epoch " << iEpoch << " of " << epochCount;

        if (monitor) {
          const int totalEpochs = getTrialEpochCount() * max(1, (int)initialWeights.size()) * learningRates.size() + getEpochCount();
          const int currentEpoch = iEpoch + (iLearningRate + iWeight * learningRates.size()) * getTrialEpochCount();
          monitor->reportProgress(100. * (currentEpoch + 1) / totalEpochs, getUpdateModel() && (iEpoch % getUpdateModel() == 0));
        }
      } /* end of epochs */

    //    newState->setAverageEpochTime(_timer.elapsed() / epochCount);

      if (iLearningRate < learningRates.size()) {
        if (iLearningRate == 0 && iWeight == 0 || !(error > bestError)) {   // using not greater instead of lesser to handle nan case.
          bestError = error;
          bestEpsilon = epsilonw;

          if (_SignalToNoiseRatio > 0) {
            // Calculate standard deviation of filters assuming zero mean.
            crbm.write_model_to_host();
            value_t filterVariance = 0;
            for (size_t iFilter = 0; iFilter < model->filter_count(); ++iFilter)
              filterVariance += dot(*model->filters()[iFilter], *model->filters()[iFilter]) / model->filters()[iFilter]->count();
            bestWeight = sqrt(filterVariance / model->filter_count()) / _SignalToNoiseRatio;
          } else {
            bestWeight = initialWeight;
          }
          dlog(Severity::Message) << "Found better learning rate: " << epsilonw << " with an error of " << bestError / X.size() << ".";
        }
      } else {
        newState->setReconstructionError(error / X.size());
        newState->setModel(model);
      }
    }
  }
}

}

}
