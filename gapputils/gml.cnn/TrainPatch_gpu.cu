/*
 * TrainPatch_gpu.cu
 *
 *  Created on: Dec 01, 2014
 *      Author: tombr
 */

#include "TrainPatch.h"

#include <tbblas/deeplearn/cnn_patches.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/random.hpp>

#include <tbblas/sequence_iterator.hpp>

namespace gml {

namespace cnn {

TrainPatchChecker::TrainPatchChecker() {
  TrainPatch test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(Mask, test);
  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(TrialEpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(PatchCounts, test);
  CHECK_MEMORY_LAYOUT2(MultiPatchCount, test);
  CHECK_MEMORY_LAYOUT2(PositiveRatio, test);

  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(LearningRates, test);
  CHECK_MEMORY_LAYOUT2(LearningDecay, test);
  CHECK_MEMORY_LAYOUT2(WeightCosts, test);
  CHECK_MEMORY_LAYOUT2(InitialWeights, test);
  CHECK_MEMORY_LAYOUT2(RandomizeTraining, test);
  CHECK_MEMORY_LAYOUT2(Model, test);
  CHECK_MEMORY_LAYOUT2(Patches, test);
  CHECK_MEMORY_LAYOUT2(Targets, test);
  CHECK_MEMORY_LAYOUT2(Predictions, test);
}

void TrainPatch::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Message);

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2> host_matrix_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  // Prepare data
  v_host_tensor_t& tensors = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  const size_t batchSize = getBatchSize();
  const size_t batchCount = tensors.size() / batchSize;
  size_t epochCount = getEpochCount();

  // Set-up patch parameters
  dim_t patchCounts = (_PatchCounts.size() == 3 ? seq(_PatchCounts[0], _PatchCounts[1], _PatchCounts[2], 1) : seq<dimCount>(1));
  dim_t patchSize = getInitialModel()->input_size() + patchCounts - 1;
  dim_t labelSize = patchCounts * seq(1, 1, 1, 0) + seq(0, 0, 0, labels[0]->size()[dimCount - 1]);
  dim_t range = tensors[0]->size() - patchSize + 1;
  dim_t patchCenter = getInitialModel()->input_size() / 2 * seq(1, 1, 1, 0);

  value_t epsilon, initialWeight = 0, bestEpsilon, bestError, bestWeight = 0;

  std::vector<double> learningRates = getLearningRates();
  std::vector<double> initialWeights = getInitialWeights();

  boost::shared_ptr<v_host_tensor_t> patches(new v_host_tensor_t());
  newState->setPatches(patches);

  boost::shared_ptr<v_host_tensor_t> targets(new v_host_tensor_t());
  newState->setTargets(targets);

  boost::shared_ptr<v_host_tensor_t> predictions(new v_host_tensor_t());
  newState->setPredictions(predictions);

  std::vector<dim_t> maskLocations, positiveLocations;
  if (getMask()) {
    host_tensor_t& mask = *getMask();
    for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (mask.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
      if (mask[*pos + patchSize / 2 * seq(1, 1, 1, 0)] > 0) {   // The center is within the lesion mask
        maskLocations.push_back(*pos);
      }
    }
  }

  for (int iWeight = 0; iWeight < initialWeights.size() || (iWeight == 0 && initialWeights.size() == 0); ++iWeight) {
    for (size_t iLearningRate = 0; iLearningRate < learningRates.size() + 1; ++iLearningRate) {

      // iLearningRate == learningRate.size() marks the final run, this is only done when all the weights were tried
      if (iLearningRate == learningRates.size() && iWeight < (int)initialWeights.size() - 1) {
        continue;
      }

      // if only one weight and one learning rate is given, do the final run immediately
      if (iWeight == 0 && iLearningRate == 0 && initialWeights.size() <= 1 && learningRates.size() == 1) {
        bestEpsilon = learningRates[0];
        if (initialWeights.size())
          bestWeight = initialWeights[0];
        continue;
      }

      if (iLearningRate < learningRates.size()) {
        epsilon = learningRates[iLearningRate];
        if (initialWeights.size())
          initialWeight = initialWeights[iWeight];
        epochCount = getTrialEpochCount();
        dlog(Severity::Message) << "Trying learning rate of " << learningRates[iLearningRate] << " and initial weight of " << initialWeight;
      } else {
        epsilon = bestEpsilon;
        if (initialWeights.size())
          initialWeight = bestWeight;
        dlog(Severity::Message) << "Final run with learning rate: " << bestEpsilon << " and initial weight of " << initialWeight;
        epochCount = getEpochCount();
      }

      boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

      if (initialWeight > 0) {
        model_t::nn_layer_t& layer = *model->nn_layers()[model->nn_layers().size() - 1];

        random_tensor2<value_t, 2, false, normal<value_t> > randn(layer.weights().size());
        tensor<value_t, 2> W = initialWeight * randn();
        layer.set_weights(W);
      }

      tbblas::deeplearn::cnn_patches<value_t, dimCount> cnn(*model, patchCounts);
      for (size_t i = 0; i < model->cnn_layers().size() && i < getFilterBatchSize().size(); ++i)
        cnn.set_batch_length(i, getFilterBatchSize()[i]);

      dlog() << "Preparation finished. Starting training.";

      value_t error, learningDecay = 1, PPV, DSC, TPR, TNR;
      tensor_t sample, label, inputPatch, labelPatch;
      host_tensor_t h_label;

      for (int iEpoch = 0; iEpoch < epochCount && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

        PPV = DSC = TPR = TNR = error = 0;

        // Learning decay only during final learning
        if (getLearningDecay() > 1 && iLearningRate == learningRates.size()) {
          learningDecay = (value_t)getLearningDecay() / ((value_t)getLearningDecay() + (value_t)iEpoch);
        }

        if (iEpoch < 10)
          momentum = initialmomentum;
        else
          momentum = finalmomentum;

        int voxelCount = 0, lesionCount = 0;

        for (int iBatch = 0; iBatch < batchCount && (monitor ? !monitor->getAbortRequested() : true); ++iBatch) {

          for (int iSample = 0; iSample < batchSize; ++iSample) {
            const int current = iSample + iBatch * batchSize;

            sample = *tensors[current];
            label = *labels[current];

            if (getPositiveRatio() >= 0.0) {
              positiveLocations.clear();
              for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (label.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
                if ((*labels[current])[*pos + patchSize / 2 * seq(1, 1, 1, 0)] > 0) {   // The center is within a lesion
                  positiveLocations.push_back(*pos);
                }
              }
            }

            value_t TP = 0, TN = 0, FP = 0, FN = 0;

            for (int iPatch = 0; iPatch < getMultiPatchCount(); ++iPatch) {
              dim_t topleft;
              if (positiveLocations.size() && (float)rand() / (float)RAND_MAX < getPositiveRatio())
                topleft = positiveLocations[rand() % positiveLocations.size()];
              else if (maskLocations.size())
                topleft = maskLocations[rand() % maskLocations.size()];
              else
                topleft = seq(rand() % range[0], rand() % range[1], rand() % range[2], 0);

              inputPatch = sample[topleft, patchSize];
              labelPatch = label[topleft + patchCenter, labelSize];

              voxelCount += labelPatch.count();
              lesionCount += sum(labelPatch > 0);

              cnn.set_input(inputPatch);
              cnn.normalize_visibles();
              cnn.update_gradient(labelPatch);

              error += sqrt(dot(labelPatch - cnn.hiddens(), labelPatch - cnn.hiddens()) / labelPatch.count()) / getMultiPatchCount();

              TP += sum((labelPatch > 0.5) * (cnn.hiddens() > 0.5));
              TN += sum((labelPatch < 0.5) * (cnn.hiddens() < 0.5));
              FP += sum((labelPatch < 0.5) * (cnn.hiddens() > 0.5));
              FN += sum((labelPatch > 0.5) * (cnn.hiddens() < 0.5));

              if (iWeight == 0 && iLearningRate == learningRates.size() && iEpoch + 1 == epochCount && iBatch + 1 == batchCount && iSample + 1 == batchSize) {
                patches->push_back(boost::make_shared<host_tensor_t>(inputPatch));
                targets->push_back(boost::make_shared<host_tensor_t>(labelPatch));
                predictions->push_back(boost::make_shared<host_tensor_t>(cnn.hiddens()));
                tbblas::synchronize();
              }
            }

            PPV += TP / (TP + FP);
            DSC += 2 * TP / (TP + FN + TP + FP);
            TPR += TP / (TP + FN);
            TNR += TN / (TN + FP);
          }

          switch (getMethod()) {
          case TrainingMethod::Momentum:
            cnn.momentum_step(epsilon * learningDecay, epsilon * learningDecay, momentum, weightcost);
            break;

          case TrainingMethod::AdaDelta:
            cnn.adadelta_step(epsilon * learningDecay, epsilon * learningDecay, momentum, weightcost);
            break;
          }
        }
//        tbblas_print((float)lesionCount / (float)voxelCount);

        dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / tensors.size()
            << " (PPV = " << PPV / tensors.size() << ", DSC = " << DSC / tensors.size() << ", TPR = " << TPR / tensors.size() << ", TNR = " << TNR / tensors.size() << ")";

        if (monitor) {
          const int totalEpochs = getTrialEpochCount() * max(1, (int)initialWeights.size()) * learningRates.size() + getEpochCount();
          const int currentEpoch = iEpoch + (iLearningRate + iWeight * learningRates.size()) * getTrialEpochCount();
          monitor->reportProgress(100 * (currentEpoch + 1) / totalEpochs);
        }
      }

      if (iLearningRate < learningRates.size()) {
        if (iLearningRate == 0 && iWeight == 0 || !(error > bestError)) {   // using not greater instead of lesser to handle nan case.
          bestError = error;
          bestEpsilon = epsilon;
          bestWeight = initialWeight;
          dlog(Severity::Message) << "Found better learning rate: " << epsilon << " with an error of " << bestError / tensors.size() << ".";
        }
      } else {
        newState->setModel(model);
      }
    }
  }
}

}

}
