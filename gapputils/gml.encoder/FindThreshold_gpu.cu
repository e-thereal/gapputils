/*
 * FindThreshold_gpu.cu
 *
 *  Created on: Jan 07, 2015
 *      Author: tombr
 */

#include "FindThreshold.h"

#include <tbblas/deeplearn/encoder.hpp>
#include "linreg.h"

//#include "optlib/BrentOptimizer.h"

namespace gml {

namespace encoder {

class Objective /* : public virtual optlib::IFunction<double> */ {
private:
  const v_host_tensor_t& labels;
  const std::vector<host_tensor_t>& predictions;
  Metric metric;

public:
  Objective(const v_host_tensor_t& labels, const std::vector<host_tensor_t>& predictions, const Metric& metric) : labels(labels), predictions(predictions), metric(metric) { }

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
      }
    }
    return metric / labels.size();
  }
};

FindThresholdChecker::FindThresholdChecker() {
  FindThreshold test;
  test.initializeClass();
  CHECK_MEMORY_LAYOUT2(InitialModel, test);
  CHECK_MEMORY_LAYOUT2(TrainingSet, test);
  CHECK_MEMORY_LAYOUT2(Labels, test);
  CHECK_MEMORY_LAYOUT2(VoxelSize, test);
  CHECK_MEMORY_LAYOUT2(TestThreshold, test);

  CHECK_MEMORY_LAYOUT2(LesionLoadsGlobal, test);
  CHECK_MEMORY_LAYOUT2(LesionLoadsTest, test);
  CHECK_MEMORY_LAYOUT2(LesionLoadsOptimal, test);
  CHECK_MEMORY_LAYOUT2(LesionLoadsPredicted, test);
  CHECK_MEMORY_LAYOUT2(PPV, test);
  CHECK_MEMORY_LAYOUT2(TPR, test);
//  CHECK_MEMORY_LAYOUT2(Model, test);
}

void FindThreshold::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  typedef tbblas::tensor<value_t, 2, true> matrix_t;
  typedef tensor_t::dim_t dim_t;

  v_host_tensor_t& data = *getTrainingSet();
  v_host_tensor_t& labels = *getLabels();

  boost::shared_ptr<data_t> lesionLoadsGlobal(new data_t(labels.size()));
  boost::shared_ptr<data_t> lesionLoadsTest(new data_t(labels.size()));
  boost::shared_ptr<data_t> lesionLoadsOptimal(new data_t(labels.size()));
  boost::shared_ptr<data_t> lesionLoadsPredicted(new data_t(labels.size()));
  boost::shared_ptr<data_t> PPVs(new data_t(labels.size()));
  boost::shared_ptr<data_t> TPRs(new data_t(labels.size()));

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

  // Get minimum and maximum lesion count
  value_t minLL, maxLL, currentLL;
  const value_t voxelVolume = _VoxelSize.prod();
  minLL = maxLL = sum(*labels[0]) * voxelVolume;

  value_t cat1, cat2, cat3, cat4, cat5;
  cat1 = cat2 = cat3 = cat4 = cat5 = 0;

  for (size_t i = 0; i < labels.size(); ++i) {
    currentLL = sum(*labels[i]) * voxelVolume;
    minLL = std::min(minLL, currentLL);
    maxLL = std::max(maxLL, currentLL);

    if (currentLL < 4000)
      ++cat1;
    else if (currentLL < 7800)
      ++cat2;
    else if (currentLL < 14700)
      ++cat3;
    else if (currentLL < 28500)
      ++cat4;
    else
      ++cat5;
  }

  dlog(Severity::Message) << "MinLL: " << minLL << ", maxLL: " << maxLL;
  dlog(Severity::Message) << "Cat1: " << cat1 << ", cat2: " << cat2 << ", cat3: " << cat3 << ", cat4: " << cat4 << ", cat5: " << cat5;

  tbblas::deeplearn::encoder<value_t, dimCount> encoder(*getInitialModel());
  std::vector<host_tensor_t> predictions;

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
      monitor->reportProgress(100. * (iSample + 1) / data.size());
    }

    predictions.push_back(label);
  }

  assert(data.size() == predictions.size());

  dlog(Severity::Message) << "Predictions calculated. Starting optimization...";

  Objective objective(labels, predictions, _Metric);
  value_t bestThreshold = 0, bestMetric = 0, currentmetric;

  for (double t = 0; t <= 1; t += 0.05) {
    currentmetric = objective.eval(t);
    dlog(Severity::Trace) << "metric at " << t << " = " << currentmetric;

    if (currentmetric > bestMetric) {
      bestMetric = currentmetric;
      bestThreshold = t;
    }
  }
  dlog(Severity::Message) << "Best threshold is " << bestThreshold;

  // Calculate metric per category
  value_t metric1, metric2, metric3, metric4, metric5, metric;
  metric1 = metric2 = metric3 = metric4 = metric5 = 0;

  for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
    host_tensor_t& label = *labels[iSample];
    const host_tensor_t& pred = predictions[iSample];

    currentLL = sum(*labels[iSample]) * voxelVolume;
    switch (_Metric) {
    case Metric::Dsc:
      metric = 2 * sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + sum(pred > bestThreshold));
      break;

    case Metric::TprPpv:
      metric = (sum((label > 0.5) * (pred > bestThreshold)) / sum(pred > bestThreshold) + sum((label > 0.5) * (pred > bestThreshold)) / sum(label > 0.5)) / 2.0;
      break;
    }
//    metric = 2 * sum((label > 0.5) * (pred > bestThreshold)) / (sum(label > 0.5) + sum(pred > bestThreshold));

    if (currentLL < 4000)
      metric1 += metric;
    else if (currentLL < 7800)
      metric2 += metric;
    else if (currentLL < 14700)
      metric3 += metric;
    else if (currentLL < 28500)
      metric4 += metric;
    else
      metric5 += metric;
  }
  dlog(Severity::Message) << "With optimal threshold: metric 1 = " << metric1 / cat1 << ", metric 2 = " << metric2 / cat2 << ", metric 3 = " << metric3 / cat3
      << ", metric 4 = " << metric4 / cat4 << ", metric 5 = " << metric5 / cat5 << ", Avg metric = " << (metric1 + metric2 + metric3 + metric4 + metric5) / labels.size();

  metric1 = metric2 = metric3 = metric4 = metric5 = 0;

  for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
    host_tensor_t& label = *labels[iSample];
    const host_tensor_t& pred = predictions[iSample];

    currentLL = sum(*labels[iSample]) * voxelVolume;
    switch (_Metric) {
    case Metric::Dsc:
      metric = 2 * sum((label > 0.5) * (pred > (value_t)_TestThreshold)) / (sum(label > 0.5) + sum(pred > (value_t)_TestThreshold));
      break;

    case Metric::TprPpv:
      metric = (sum((label > 0.5) * (pred > (value_t)_TestThreshold)) / sum(pred > (value_t)_TestThreshold) + sum((label > 0.5) * (pred > (value_t)_TestThreshold)) / sum(label > 0.5)) / 2.0;
      break;
    }
//    metric = 2 * sum((label > 0.5) * (pred > (value_t)_TestThreshold)) / (sum(label > 0.5) + sum(pred > (value_t)_TestThreshold));

    if (currentLL < 4000)
      metric1 += metric;
    else if (currentLL < 7800)
      metric2 += metric;
    else if (currentLL < 14700)
      metric3 += metric;
    else if (currentLL < 28500)
      metric4 += metric;
    else
      metric5 += metric;
  }
  dlog(Severity::Message) << "With test threshold: metric 1 = " << metric1 / cat1 << ", metric 2 = " << metric2 / cat2 << ", metric 3 = " << metric3 / cat3 << ", metric 4 = " << metric4 / cat4 << ", metric 5 = " << metric5 / cat5 << ", Avg metric = " << (metric1 + metric2 + metric3 + metric4 + metric5) / labels.size();
  assert(cat1 + cat2 + cat3 + cat4 + cat5 == labels.size());

  // Find optimal threshold
  LinearRegression adaThreshold;
  value_t currentBestThreshold, predictedLL;

  metric1 = metric2 = metric3 = metric4 = metric5 = 0;

  for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
    host_tensor_t& label = *labels[iSample];
    const host_tensor_t& pred = predictions[iSample];

    currentLL = sum(*labels[iSample]) * voxelVolume;

    // Predict lesion load with best threshold
    predictedLL = sum(pred > bestThreshold) * voxelVolume;

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
      }
//      metric = 2 * sum((label > 0.5) * (pred > t)) / (sum(label > 0.5) + sum(pred > t));
      if (metric > bestMetric) {
        bestMetric = metric;
        currentBestThreshold = t;
      }
    }

    if (currentLL < 4000)
      metric1 += bestMetric;
    else if (currentLL < 7800)
      metric2 += bestMetric;
    else if (currentLL < 14700)
      metric3 += bestMetric;
    else if (currentLL < 28500)
      metric4 += bestMetric;
    else
      metric5 += bestMetric;

    // Add both to the regression model
    adaThreshold.addXY(predictedLL, currentBestThreshold);

    lesionLoadsGlobal->at(iSample) = predictedLL;
    lesionLoadsTest->at(iSample) = sum(pred > _TestThreshold) * voxelVolume;
    lesionLoadsOptimal->at(iSample) = sum(pred > currentBestThreshold) * voxelVolume;
  }

  dlog(Severity::Message) << "With optimal thresholds: metric 1 = " << metric1 / cat1 << ", metric 2 = " << metric2 / cat2 << ", metric 3 = " << metric3 / cat3 << ", metric 4 = " << metric4 / cat4 << ", metric 5 = " << metric5 / cat5 << ", Avg metric = " << (metric1 + metric2 + metric3 + metric4 + metric5) / labels.size();

  // Run analysis with predicted thresholds
  metric1 = metric2 = metric3 = metric4 = metric5 = 0;

  for (size_t iSample = 0; iSample < labels.size(); ++iSample) {
    host_tensor_t& label = *labels[iSample];
    const host_tensor_t& pred = predictions[iSample];

    currentLL = sum(*labels[iSample]) * voxelVolume;
    predictedLL = sum(pred > bestThreshold) * voxelVolume;

    const value_t threshold = adaThreshold.estimateY(predictedLL);
    switch (_Metric) {
    case Metric::Dsc:
      metric = 2 * sum((label > 0.5) * (pred > threshold)) / (sum(label > 0.5) + sum(pred > threshold));
      break;

    case Metric::TprPpv:
      metric = (sum((label > 0.5) * (pred > threshold)) / sum(pred > threshold) + sum((label > 0.5) * (pred > threshold)) / sum(label > 0.5)) / 2.0;
      break;
    }
//    metric = 2 * sum((label > 0.5) * (pred > threshold)) / (sum(label > 0.5) + sum(pred > threshold));

    if (currentLL < 4000)
      metric1 += metric;
    else if (currentLL < 7800)
      metric2 += metric;
    else if (currentLL < 14700)
      metric3 += metric;
    else if (currentLL < 28500)
      metric4 += metric;
    else
      metric5 += metric;

    lesionLoadsPredicted->at(iSample) = sum(pred > threshold) * voxelVolume;

    PPVs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / sum(pred > bestThreshold);
    TPRs->at(iSample) = sum((label > 0.5) * (pred > bestThreshold)) / sum(label > 0.5);
  }
  dlog(Severity::Message) << "With predicted thresholds: metric 1 = " << metric1 / cat1 << ", metric 2 = " << metric2 / cat2 << ", metric 3 = " << metric3 / cat3 << ", metric 4 = " << metric4 / cat4 << ", metric 5 = " << metric5 / cat5 << ", Avg metric = " << (metric1 + metric2 + metric3 + metric4 + metric5) / labels.size();

  newState->setLesionLoadsGlobal(lesionLoadsGlobal);
  newState->setLesionLoadsTest(lesionLoadsTest);
  newState->setLesionLoadsOptimal(lesionLoadsOptimal);
  newState->setLesionLoadsPredicted(lesionLoadsPredicted);

  newState->setPPV(PPVs);
  newState->setTPR(TPRs);

//  optlib::BrentOptimizer optimizer;
//  optimizer.setStepSize(0.2);
//  optimizer.setTolerance(0.01);
//
//  double threshold = 0.2;
//  optimizer.maximize(threshold, objective);
//
//
//  dlog(Severity::Message) << "Optimal threshold = " << threshold << " at a metric of " << objective.eval(threshold);
}

}

}
