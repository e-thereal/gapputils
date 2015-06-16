/*
 * TrainPatch_gpu.cu
 *
 *  Created on: Aug 14, 2014
 *      Author: tombr
 */

#include "TrainPatch.h"

#include <tbblas/deeplearn/nn.hpp>
#include <tbblas/io.hpp>
#include <tbblas/sum.hpp>
#include <tbblas/dot.hpp>
#include <tbblas/reshape.hpp>
#include <tbblas/ones.hpp>

#include <tbblas/sequence_iterator.hpp>
#include <set>
#include <algorithm>

namespace gml {

namespace nn {

TrainPatchChecker::TrainPatchChecker() {
  TrainPatch test;
  test.initializeClass();

  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(Mask, test);
  CHECK_MEMORY_LAYOUT2(PatchWidth, test);
  CHECK_MEMORY_LAYOUT2(PatchHeight, test);
  CHECK_MEMORY_LAYOUT2(PatchDepth, test);
  CHECK_MEMORY_LAYOUT2(PatchCount, test);
  CHECK_MEMORY_LAYOUT2(SelectionMethod, test);
  CHECK_MEMORY_LAYOUT2(PositiveRatio, test);
  CHECK_MEMORY_LAYOUT2(MinimumBucketSizes, test);
  CHECK_MEMORY_LAYOUT2(BucketRatio, test);

  CHECK_MEMORY_LAYOUT2(EpochCount, test);
  CHECK_MEMORY_LAYOUT2(BatchSize, test);
  CHECK_MEMORY_LAYOUT2(Objective, test);
  CHECK_MEMORY_LAYOUT2(SensitivityRatio, test);
  CHECK_MEMORY_LAYOUT2(Method, test);
  CHECK_MEMORY_LAYOUT2(LearningRate, test);
  CHECK_MEMORY_LAYOUT2(WeightCosts, test);
  CHECK_MEMORY_LAYOUT2(DropoutRates, test);
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

  const int dimCount = host_tensor_t::dimCount;

  typedef int8_t bucket_id_t;
  typedef model_t::value_t value_t;
  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tbblas::tensor<value_t, 2, false> host_matrix_t;

  typedef tbblas::tensor<value_t, dimCount, true> tensor_t;
  typedef tbblas::tensor<bucket_id_t, dimCount, false> bucket_id_tensor_t;
  typedef tensor_t::dim_t dim_t;

  if (getTrainingSet()->size() != getLabels()->size()) {
    dlog(Severity::Warning) << "The sizes of the training and label set don't match. Aborting!";
    return;
  }

  if (getSelectionMethod() == PatchSelectionMethod::LeitnerSystem && getPositiveRatio() <= 0) {
    dlog(Severity::Warning) << "The positive ratio must be greater than 0 to use the Leitner system. Aborting!";
    return;
  }

  if (getSelectionMethod() == PatchSelectionMethod::LeitnerSystem && _MinimumBucketSizes.size() < 2) {
    dlog(Severity::Warning) << "You need at least 2 buckets in order to use the Leitner system. Aborting!";
    return;
  }

  boost::shared_ptr<model_t> model(new model_t(*getInitialModel()));

  tbblas::deeplearn::nn<value_t> nn(*model);
  nn.set_objective_function(getObjective());
  nn.set_sensitivity_ratio(getSensitivityRatio());

  for (size_t i = 0; i < model->layers().size() && i < _DropoutRates.size(); ++i) {
    nn.set_dropout_rate(i, _DropoutRates[i]);
  }

  nn.visibles().resize(seq((int)getBatchSize() * getPatchCount(), (int)model->visibles_count()));

  // Prepare data
  v_host_tensor_t& data = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  if (getPatchWidth() * getPatchHeight() * getPatchDepth() * data[0]->size()[dimCount - 1] != model->visibles_count()) {
    dlog(Severity::Warning) << "Patch dimension doesn't match the number of visible units of the neural network. Aborting!";
    return;
  }

  if (getPositiveRatio() >= 0.0 && labels[0]->size()[dimCount - 1] != 1) {
    dlog(Severity::Warning) << "Positive ratio can only be used for binary classification (channels of the label image must be 1). Aborting!";
    return;
  }

  dim_t patchSize = seq(getPatchWidth(), getPatchHeight(), getPatchDepth(), data[0]->size()[dimCount - 1]);
  dim_t labelSize = seq(1, 1, 1, labels[0]->size()[dimCount - 1]);
  dim_t range = data[0]->size() - patchSize + 1;
  dim_t patchCenter = patchSize / 2 * seq(1,1,1,0);

  matrix_t yBatch(getBatchSize() * getPatchCount(), model->hiddens_count());
  host_matrix_t h_predictions(yBatch.size()), h_targets(yBatch.size());

  matrix_t res;
  value_t weightcost = getWeightCosts();
  value_t initialmomentum = 0.5f;
  value_t finalmomentum = 0.9f;
  value_t momentum;

  const int batchSize = getBatchSize();
  const int batchCount = data.size() / batchSize;

  value_t error, PPV, DSC, TPR, TNR, ACC;
  tensor_t tensor, label;
  host_tensor_t h_label;

  boost::shared_ptr<v_host_tensor_t> patches(new v_host_tensor_t());
  newState->setPatches(patches);

  boost::shared_ptr<v_host_tensor_t> targets(new v_host_tensor_t());
  newState->setTargets(targets);

  boost::shared_ptr<v_host_tensor_t> predictions(new v_host_tensor_t());
  newState->setPredictions(predictions);

  // bucket IDs
  // -1: ignore
  //  0: unknown
  //  1: Bucket one (hard cases)
  //  2: Bucket two (correctly solved once)
  //  3: Bucket three (correctly solved twice)
  //  4: Bucket four (correctly solved four times)

  // At the beginning, bucket one will always be refilled with random cases to contain at least the minimum number of samples
  // This is temporarily. Filled samples are not tagged as belonging to bucket one.

  std::vector<bucket_id_tensor_t> bucketIds;

  std::vector<dim_t> positiveLocations, negativeLocations, maskLocations, selectedLocations(getPatchCount() * getBatchSize());
  std::vector<std::vector<dim_t> > bucketLocations(_MinimumBucketSizes.size());
  std::vector<size_t> selectedSamples(getBatchSize()), positivesPerBucket(_MinimumBucketSizes.size());

  if (getSelectionMethod() == PatchSelectionMethod::LeitnerSystem) {
    bucketIds.resize(labels.size());

    if (getMask()) {
      host_tensor_t& mask = *getMask();
      for (size_t iSample = 0; iSample < bucketIds.size(); ++iSample) {
        host_tensor_t& label = *labels[iSample];
        // TODO: uncomment this line
//        bucketIds[iSample] = -1 * ones<value_t>(label.size()) + (mask > 0);
      }
    } else {
      for (size_t iSample = 0; iSample < bucketIds.size(); ++iSample) {
        host_tensor_t& label = *labels[iSample];
        bucketIds[iSample] = zeros<bucket_id_t>(label.size());
      }
    }
  } else {
    if (getMask()) {
      host_tensor_t& mask = *getMask();
      for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (mask.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
        if (mask[*pos + patchSize / 2 * seq(1, 1, 1, 0)] > 0) {   // The center is within the ROI mask
          maskLocations.push_back(*pos);
        }
      }
    }
  }

  dlog() << "Preparation finished. Starting training.";

  for (int iEpoch = 0; iEpoch < getEpochCount() && (monitor ? !monitor->getAbortRequested() : true); ++iEpoch) {

    ACC = PPV = DSC = TPR = TNR = error = 0;

    if (iEpoch < 10)
      momentum = initialmomentum;
    else
      momentum = finalmomentum;

    for (int iBatch = 0; iBatch < batchCount; ++iBatch) {

      for (int iSample = 0; iSample < batchSize; ++iSample) {
        // Fill batch with random patches
        tensor = *data[iBatch * batchSize + iSample];
        label = *labels[iBatch * batchSize + iSample];
        h_label = *labels[iBatch * batchSize + iSample];
        tbblas::synchronize();

        selectedSamples[iSample] = iBatch * batchSize + iSample;

        int positivePatchCount = 0;

        if (getSelectionMethod() == PatchSelectionMethod::LeitnerSystem) {
          positiveLocations.clear();
          negativeLocations.clear();

          for (size_t iBucket = 0; iBucket < bucketLocations.size(); ++iBucket) {
            bucketLocations[iBucket].clear();
            positivesPerBucket[iBucket] = 0;
          }

          // Fill buckets with potential locations
          // Also select unknown positive and negative samples for quicker refilling the first bucket if needed

          for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (label.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
            const dim_t center = *pos + patchSize / 2 * seq(1, 1, 1, 0);  // center of the patch
            const bucket_id_t bucketId = bucketIds[iBatch * batchSize + iSample][center];

            if (bucketId > 0 && bucketId <= bucketLocations.size()) {
              bucketLocations[bucketId - 1].push_back(*pos);
              positivesPerBucket[bucketId - 1] += h_label[center] > 0;
            } else if (bucketId == 0) {
              if (h_label[center] > 0) {    // The center belongs to the positive class
                positiveLocations.push_back(*pos);
              } else {
                negativeLocations.push_back(*pos);
              }
            }
          }

          if (iSample == 0 && iBatch == 0) {
            LogEntry entry = dlog(Severity::Trace);

            // Get number of positive patches per bucket

            entry << "iBatch = " << iBatch << ", iSample = " << iSample << ", Buckets: " << positivesPerBucket[0] << "/" << bucketLocations[0].size();
            for (size_t i = 1; i < bucketLocations.size(); ++i)
              entry << ", " << positivesPerBucket[i] << "/" << bucketLocations[i].size();
          }

          // Faster drawing samples by shuffling the list and then go from the top
          std::random_shuffle(positiveLocations.begin(), positiveLocations.end());
          std::random_shuffle(negativeLocations.begin(), negativeLocations.end());

          for (size_t iPositive = 0, iNegative = 0; bucketLocations[0].size() < _MinimumBucketSizes[0];) {
            if (iPositive < positiveLocations.size() && iNegative < negativeLocations.size()) {
              if ((float)rand() / (float)RAND_MAX < getPositiveRatio()) {
                bucketLocations[0].push_back(positiveLocations[iPositive++]);
              } else {
                bucketLocations[0].push_back(negativeLocations[iNegative++]);
              }
            } else if (iPositive < positiveLocations.size()) {
              bucketLocations[0].push_back(positiveLocations[iPositive++]);
            } else if (iNegative < negativeLocations.size()) {
              bucketLocations[0].push_back(negativeLocations[iNegative++]);
            } else {
              break;
            }
          }

          // Randomize bucket locations, so I can just iterate throw the location vector to get unique random samples
          for (size_t iBucket = 0; iBucket < bucketLocations.size(); ++iBucket) {
            std::random_shuffle(bucketLocations[iBucket].begin(), bucketLocations[iBucket].end());
          }

        } else {
          if (getPositiveRatio() >= 0) {
            positiveLocations.clear();
            for (sequence_iterator<dim_t> pos(seq<dimCount>(0), (label.size() - patchSize) * seq(1, 1, 1, 0) + seq(0, 0, 0, 1)); pos; ++pos) {
              if (h_label[*pos + patchSize / 2 * seq(1, 1, 1, 0)] > 0) {   // The center belongs to the positive class
                positiveLocations.push_back(*pos);
              }
            }
          }
        }

        std::vector<size_t> iPos(bucketLocations.size(), 0);

        for (int iPatch = 0; iPatch < getPatchCount(); ++iPatch) {
          dim_t topleft;

          if (getSelectionMethod() == PatchSelectionMethod::LeitnerSystem) {

            // Initialize the topleft with a patch location from the first bucket
            for (size_t iBucket = 0; iBucket < bucketLocations.size(); ++iBucket) {
              if (iPos[iBucket] < bucketLocations[iBucket].size()) {
                topleft = bucketLocations[iBucket][iPos[iBucket]++];
                break;
              }
            }

            // If we don't want a sample from the first bucket, try the next buckets
            // If a randomly selected bucket doesn't fulfill the size requirements, fall back to the first bucket (keep initial value)
            // If no bucket was randomly selected, also fall back to the first bucket (keep initial value)
            if ((float)rand() / (float)RAND_MAX > getBucketRatio()) {
              for (size_t iBucket = 1; iBucket < bucketLocations.size(); ++iBucket) {
                if ((float)rand() / (float)RAND_MAX < getBucketRatio()) {
                  if (bucketLocations[iBucket].size() >= _MinimumBucketSizes[iBucket] && iPos[iBucket] < bucketLocations[iBucket].size()) {
                    topleft = bucketLocations[iBucket][iPos[iBucket]++];
                  }
                  break;
                }
              }
            }

            selectedLocations[iSample * getPatchCount() + iPatch] = topleft;
          } else {
            if (positiveLocations.size() && (float)rand() / (float)RAND_MAX < getPositiveRatio())
              topleft = positiveLocations[rand() % positiveLocations.size()];
            else if (maskLocations.size())
              topleft = maskLocations[rand() % maskLocations.size()];
            else
              topleft = seq(rand() % range[0], rand() % range[1], rand() % range[2], 0);
          }

          if (h_label[topleft + patchCenter] > 0.5)
            ++positivePatchCount;

          // Fill visible units and targets
          row(nn.visibles(), iSample * getPatchCount() + iPatch) = reshape(tensor[topleft, patchSize], 1, model->visibles_count());
          row(yBatch, iSample * getPatchCount() + iPatch) = reshape(label[topleft + patchCenter, labelSize], 1, model->hiddens_count());

          if (iEpoch + 1 == getEpochCount() && iBatch + 1 == batchCount && iSample + 1 == batchSize) {
            patches->push_back(boost::make_shared<host_tensor_t>(tensor[topleft, patchSize]));
            targets->push_back(boost::make_shared<host_tensor_t>(label[topleft + patchCenter, labelSize]));
          }
        }
        if (iSample == 0 && iBatch == 0) {
          dlog(Severity::Trace) << "Positive patches = " << positivePatchCount << "; total number of patches = " << getPatchCount();
//          std::cout << "iPos: " << iPos[0];
//          for (size_t i = 1; i < iPos.size(); ++i)
//            std::cout << ", " << iPos[i];
//          std::cout << std::endl;
        }
      }

      // Update model
      // TODO: update code to reflect latest trainer changes
//      switch (getMethod()) {
//      case TrainingMethod::Momentum:
//        nn.momentum_update(yBatch, getLearningRate(), momentum, weightcost);
//        break;
//
//      case TrainingMethod::AdaDelta:
//        nn.adadelta_update(yBatch, getLearningRate(), momentum, weightcost);
//        break;
//      }
//
//      // Calculate errors
//      switch (getObjective()) {
//      case tbblas::deeplearn::objective_function::SSD:
//      case tbblas::deeplearn::objective_function::SenSpe:
//        error += sqrt(dot(nn.hiddens() - yBatch, nn.hiddens() - yBatch) / yBatch.size()[0]);
//        break;
//
//      case tbblas::deeplearn::objective_function::DSC:
//        error += 2 * sum(nn.hiddens() * yBatch) / (sum(nn.hiddens()) + sum(yBatch));
//        break;
//
//      case tbblas::deeplearn::objective_function::DSC2:
//        error += 2 * sum((value_t(1) + value_t(-1) * (nn.hiddens() - yBatch) * (nn.hiddens() - yBatch)) * yBatch) / (sum(nn.hiddens() * nn.hiddens()) + sum(yBatch));
//        break;
//      }

      if (iEpoch + 1 == getEpochCount() && iBatch + 1 == batchCount) {
        for (size_t iPatch = 0; iPatch < getPatchCount(); ++iPatch)
          predictions->push_back(boost::make_shared<host_tensor_t>(reshape(row(nn.hiddens(), (batchSize - 1) * getPatchCount() + iPatch), labelSize)));
      }

      PPV += sum((yBatch > 0.5) * (nn.hiddens() > 0.5)) / sum(nn.hiddens() > 0.5);
      DSC += 2 * sum ((yBatch > 0.5) * (nn.hiddens() > 0.5)) / (sum(yBatch > 0.5) + sum(nn.hiddens() > 0.5));
      TPR += sum((yBatch > 0.5) * (nn.hiddens() > 0.5)) / sum(yBatch > 0.5);
      TNR += sum((yBatch < 0.5) * (nn.hiddens() < 0.5)) / sum(yBatch < 0.5);
      ACC += sum((yBatch > 0.5) == (nn.hiddens() > 0.5)) / (value_t)yBatch.count();

      if (getSelectionMethod() == PatchSelectionMethod::LeitnerSystem) {
        // Figure out if patches were classified correctly and update the bucketIds
        h_predictions = nn.hiddens();
        h_targets = yBatch;
        tbblas::synchronize();

        for (size_t iPatch = 0; iPatch < selectedLocations.size(); ++iPatch) {
          const size_t iSample = selectedSamples[iPatch / getPatchCount()];
          bucket_id_t& bucketPatchId = bucketIds[iSample][selectedLocations[iPatch] + patchSize / 2 * seq(1, 1, 1, 0)];
          if ((h_predictions.data()[iPatch] > 0.5) == (h_targets.data()[iPatch] > 0.5)) {
            if (bucketPatchId == 0) {
              bucketPatchId = 2;
            } else if (bucketPatchId < bucketLocations.size()) {
              ++bucketPatchId;
            }
          } else {
            bucketPatchId = 1;
          }
        }
      }
    }

    dlog(Severity::Trace) << "Error at epoch " << iEpoch + 1 << " of " << getEpochCount() << " epochs: " << error / batchCount
        << " (PPV = " << PPV / batchCount << ", DSC = " << DSC / batchCount << ", TPR = " << TPR / batchCount << ", TNR = " << TNR / batchCount << ", ACC = " << ACC / batchCount << ")";

    if (monitor)
      monitor->reportProgress(100. * (iEpoch + 1) / getEpochCount());
  }

  newState->setModel(model);
}

}

}
