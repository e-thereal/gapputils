/*
 * TestThreshold_gpu.cu
 *
 *  Created on: Jan 22, 2015
 *      Author: tombr
 */

#include "TestThreshold.h"

#include <tbblas/deeplearn/encoder.hpp>
#include "linreg.h"

namespace gml {

namespace encoder {

class TestObjective {
private:
  const v_host_tensor_t& labels;
  const std::vector<host_tensor_t>& predictions;
  Metric metric;

public:
  TestObjective(const v_host_tensor_t& labels, const std::vector<host_tensor_t>& predictions, const Metric& metric) : labels(labels), predictions(predictions), metric(metric) { }

  virtual double eval(const double& value) {
    using namespace tbblas;

    double metric = 0;
    const float threshold = value;
    for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
      host_tensor_t& label = *labels[iSample];
      const host_tensor_t& pred = predictions[iSample];

      switch (this->metric) {
      case Metric::Dsc:
        metric += 2 * sum((label > 0.5) * (pred > threshold)) / (sum(label > 0.5) + sum(pred > threshold));
        break;

      case Metric::TprPpv:
        metric += (sum((label > 0.5) * (pred > threshold)) / sum(pred > threshold) + sum((label > 0.5) * (pred > threshold)) / sum(label > 0.5)) / 2.0;
        break;

      case Metric::MinTprPpv:
        metric += min(sum((label > 0.5) * (pred > threshold)) / sum(pred > threshold), sum((label > 0.5) * (pred > threshold)) / sum(label > 0.5));
        break;
      }
    }
    return metric / labels.size();
  }
};

TestThresholdChecker::TestThresholdChecker() {
  TestThreshold test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(TrainingLabels, test);
  CHECK_MEMORY_LAYOUT2(TestSet, test);
  CHECK_MEMORY_LAYOUT2(TestLabels, test);
  CHECK_MEMORY_LAYOUT2(Metric, test);
  CHECK_MEMORY_LAYOUT2(FilterBatchSize, test);
  CHECK_MEMORY_LAYOUT2(SubRegionCount, test);
  CHECK_MEMORY_LAYOUT2(GlobalTPR, test);
  CHECK_MEMORY_LAYOUT2(GlobalPPV, test);
  CHECK_MEMORY_LAYOUT2(GlobalDSC, test);
  CHECK_MEMORY_LAYOUT2(OptimalTPR, test);
  CHECK_MEMORY_LAYOUT2(OptimalPPV, test);
  CHECK_MEMORY_LAYOUT2(OptimalDSC, test);
  CHECK_MEMORY_LAYOUT2(PredictedTPR, test);
  CHECK_MEMORY_LAYOUT2(PredictedPPV, test);
  CHECK_MEMORY_LAYOUT2(PredictedDSC, test);
}

void TestThreshold::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tensor_t::dim_t dim_t;

  v_host_tensor_t& data = *getTrainingSet();
  v_host_tensor_t& labels = *getTrainingLabels();

  v_host_tensor_t& testData = *getTestSet();
  v_host_tensor_t& testLabels = *getTestLabels();

  boost::shared_ptr<data_t> gTPRs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> gPPVs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> gDSCs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> oTPRs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> oPPVs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> oDSCs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> pTPRs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> pPPVs(new data_t(testLabels.size()));
  boost::shared_ptr<data_t> pDSCs(new data_t(testLabels.size()));

  if (data[0]->size() != getInitialModel()->inputs_size()) {
    dlog(Severity::Warning) << "Input size does not match the size of visible units of the encoder network. Aborting!";
    return;
  }

  if (labels[0]->size() != getInitialModel()->outputs_size()) {
    dlog(Severity::Warning) << "Label size does not match the size of visible units of the encoder network. Aborting!";
    return;
  }

  if (data.size() != labels.size()) {
    dlog(Severity::Warning) << "Need to have same number of input samples and labels. Aborting!";
    return;
  }

  if (data[0]->size() != testData[0]->size() || labels[0]->size() != testLabels[0]->size() || testData.size() != testLabels.size()) {
    dlog(Severity::Warning) << "Invalid test data. Aborting!";
    return;
  }

  // Get minimum and maximum lesion count
  tbblas::deeplearn::encoder<value_t, dimCount> encoder(*getInitialModel(), _SubRegionCount);
  for (size_t i = 0; i < getInitialModel()->cnn_encoders().size() + getInitialModel()->cnn_decoders().size() && i < getFilterBatchSize().size(); ++i)
    encoder.set_batch_length(i, getFilterBatchSize()[i]);

  std::vector<host_tensor_t> predictions, testPredictions;

  tensor_t sample;
  host_tensor_t label;
  for (size_t iSample = 0; iSample < data.size() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
    sample = *data[iSample];

    encoder.inputs() = sample;

    // Perform forward propagation
    encoder.infer_outputs();
    label = encoder.outputs();
    tbblas::synchronize();

    if (monitor) {
      monitor->reportProgress(100. * (iSample + 1) / (data.size() + testData.size()));
    }

    predictions.push_back(label);
  }

  for (size_t iSample = 0; iSample < testData.size() && (monitor ? !monitor->getAbortRequested() : true); ++iSample) {
    sample = *testData[iSample];

    encoder.inputs() = sample;

    // Perform forward propagation
    encoder.infer_outputs();
    label = encoder.outputs();
    tbblas::synchronize();

    if (monitor) {
      monitor->reportProgress(100. * (iSample + data.size() + 1) / (data.size() + testData.size()));
    }

    testPredictions.push_back(label);
  }

  assert(data.size() == predictions.size());
  assert(testData.size() == testPredictions.size());

  dlog(Severity::Message) << "Predictions calculated. Starting optimization...";

  TestObjective objective(labels, predictions, _Metric);
  value_t bestThreshold = 0, bestMetric = 0, currentMetric, metric;

  for (double t = 0; t <= 1; t += 0.05) {
    currentMetric = objective.eval(t);
    dlog(Severity::Trace) << "metric at " << t << " = " << currentMetric;

    if (currentMetric > bestMetric) {
      bestMetric = currentMetric;
      bestThreshold = t;
    }
  }
  dlog(Severity::Message) << "Best threshold is " << bestThreshold << " (" << _Metric << " = " << bestMetric << ")";

  value_t meanTPR, meanPPV, meanDSC;

  meanTPR = meanPPV = meanDSC = 0;

  // Calculate performance using a global threshold
  for (size_t iSample = 0; iSample < testData.size(); ++iSample) {
    host_tensor_t& label = *testLabels[iSample];
    const host_tensor_t& pred = testPredictions[iSample];

    meanTPR += gTPRs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / sum(label > 0.5);
    meanPPV += gPPVs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / sum(pred > bestThreshold);
    meanDSC += gDSCs->at(iSample) = 2 * sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + sum(pred > bestThreshold));
  }

  dlog(Severity::Message) << "Global threshold: TPR = " << meanTPR / testData.size() << ", PPV = " << meanPPV / testData.size() << ", DSC = " << meanDSC / testData.size();

  meanTPR = meanPPV = meanDSC = 0;

  // Calculate the performance using an optimal threshold
  value_t currentBestThreshold;
  for (size_t iSample = 0; iSample < testData.size(); ++iSample) {
    host_tensor_t& label = *testLabels[iSample];
    const host_tensor_t& pred = testPredictions[iSample];

    // Find optimal threshold for this example
    currentBestThreshold = bestMetric = 0;
    for (value_t t = 0; t <= 1; t += 0.05) {
      switch (_Metric) {
      case Metric::Dsc:
        metric = 2 * sum((label > 0.5) * (pred > t)) / (sum(label > 0.5) + sum(pred > t));
        break;

      case Metric::TprPpv:
        metric = (sum((label > 0.5) * (pred > t)) / sum(pred > t) + sum((label > 0.5) * (pred > t)) / sum(label > 0.5)) / 2.0;
        break;

      case Metric::MinTprPpv:
        metric = min(sum((label > 0.5) * (pred > t)) / sum(pred > t), sum((label > 0.5) * (pred > t)) / sum(label > 0.5));
        break;
      }

      if (metric > bestMetric) {
        bestMetric = metric;
        currentBestThreshold = t;
      }
    }

    meanTPR += oTPRs->at(iSample) = sum((label > 0.5) * (pred > currentBestThreshold)) / sum(label > 0.5);
    meanPPV += oPPVs->at(iSample) = sum((label > 0.5) * (pred > currentBestThreshold)) / sum(pred > currentBestThreshold);
    meanDSC += oDSCs->at(iSample) = 2 * sum((label > 0.5) * (pred > currentBestThreshold)) / (sum(label > 0.5) + sum(pred > currentBestThreshold));
  }

  dlog(Severity::Message) << "Optimal threshold: TPR = " << meanTPR / testData.size() << ", PPV = " << meanPPV / testData.size() << ", DSC = " << meanDSC / testData.size();

  // Calculate the performance using a predicted threshold

  // Find optimal threshold
  LinearRegression adaThreshold;
  value_t predictedLL;

  for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
    host_tensor_t& label = *labels[iSample];
    const host_tensor_t& pred = predictions[iSample];

    // Predict lesion load with best threshold
    predictedLL = sum(pred > bestThreshold);

    // Find optimal threshold for this example
    currentBestThreshold = bestMetric = 0;
    for (value_t t = 0; t <= 1; t += 0.05) {
      switch (_Metric) {
      case Metric::Dsc:
        metric = 2 * sum((label > 0.5) * (pred > t)) / (sum(label > 0.5) + sum(pred > t));
        break;

      case Metric::TprPpv:
        metric = (sum((label > 0.5) * (pred > t)) / sum(pred > t) + sum((label > 0.5) * (pred > t)) / sum(label > 0.5)) / 2.0;
        break;

      case Metric::MinTprPpv:
        metric = min(sum((label > 0.5) * (pred > t)) / sum(pred > t), sum((label > 0.5) * (pred > t)) / sum(label > 0.5));
        break;
      }

      if (metric > bestMetric) {
        bestMetric = metric;
        currentBestThreshold = t;
      }
    }

    // Add both to the regression model
    adaThreshold.addXY(predictedLL, currentBestThreshold);
  }

  meanTPR = meanPPV = meanDSC = 0;

  // Run analysis with predicted thresholds
  for (size_t iSample = 0; iSample < testLabels.size(); ++iSample) {
    host_tensor_t& label = *testLabels[iSample];
    const host_tensor_t& pred = testPredictions[iSample];

    predictedLL = sum(pred > bestThreshold);

    const value_t threshold = adaThreshold.estimateY(predictedLL);

    meanTPR += pTPRs->at(iSample) = sum((label > 0.5) * (pred > threshold)) / sum(label > 0.5);
    meanPPV += pPPVs->at(iSample) = sum((label > 0.5) * (pred > threshold)) / sum(pred > threshold);
    meanDSC += pDSCs->at(iSample) = 2 * sum((label > 0.5) * (pred > threshold)) / (sum(label > 0.5) + sum(pred > threshold));
  }

  dlog(Severity::Message) << "Predicted threshold: TPR = " << meanTPR / testData.size() << ", PPV = " << meanPPV / testData.size() << ", DSC = " << meanDSC / testData.size();

  newState->setGlobalTPR(gTPRs);
  newState->setGlobalPPV(gPPVs);
  newState->setGlobalDSC(gDSCs);
  newState->setOptimalTPR(oTPRs);
  newState->setOptimalPPV(oPPVs);
  newState->setOptimalDSC(oDSCs);
  newState->setPredictedTPR(pTPRs);
  newState->setPredictedPPV(pPPVs);
  newState->setPredictedDSC(pDSCs);
}

}

}
