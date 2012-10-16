/*
 * Trainer.cpp
 *
 *  Created on: Oct 4, 2012
 *      Author: tombr
 */

#include "Trainer.h"
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>

#include <algorithm>
#include <fstream>
#include <capputils/EnumeratorAttribute.h>
#include <capputils/FilenameAttribute.h>

#include <gpusvm/svmTrain.h>

namespace gapputils {
namespace ml {
namespace segmentation {

using namespace capputils::attributes;
using namespace gapputils::attributes;

BeginPropertyDefinitions(Trainer)

  ReflectableBase(workflow::DefaultWorkflowElement<Trainer>)

  WorkflowProperty(FeatureMaps, Input("F"), NotNull<Type>())
  WorkflowProperty(Segmentations, Input("Seg"), NotNull<Type>())
  WorkflowProperty(ModelName, Filename(), NotEmpty<Type>("Not a valid filename."))
  WorkflowProperty(Rank, Description("Number of support vectors."))
  WorkflowProperty(Min)
  WorkflowProperty(MaxGamma)
  WorkflowProperty(MaxC)
  WorkflowProperty(Steps)
  WorkflowProperty(Tolerance)
  WorkflowProperty(MaxIterations)
  WorkflowProperty(CvFolds, Description("Number of folds used for k-fold cross validation."))
  WorkflowProperty(CvImageCount, Description("Number of images used to estimate hyperparameters using cross validation. A count of -1 denotes that all images should be used."))
  WorkflowProperty(RandomizeSamples)
  WorkflowProperty(OutputName, Output("SVC"), Filename(), Description("Will be set to ModelName after the update."))

EndPropertyDefinitions

Trainer::Trainer() : _Rank(10), _Min(1e-5), _MaxGamma(1), _MaxC(1e5), _Steps(4), _Tolerance(0.01), _MaxIterations(10),
_CvFolds(4), _CvImageCount(1), _RandomizeSamples(true)
{
  setLabel("Trainer");
}

Trainer::~Trainer() { }

using namespace capputils;
using namespace dlib;

typedef double value_t;
typedef matrix<value_t, 0, 1> sample_t;
typedef radial_basis_kernel<sample_t> kernel_t;

struct objective {

  objective(const std::vector<sample_t>& samples,
      const std::vector<value_t>& labels,
      const Trainer& trainer)
  : samples(samples), labels(labels), trainer(trainer), best_value(0) { }

  value_t operator()(const matrix<value_t>& params) const {
    const value_t gamma = exp(params(0));
    const value_t c = exp(params(1));

    svm_c_trainer<kernel_t> trainer;
    trainer.set_kernel(kernel_t(gamma));
    trainer.set_c(c);

    matrix<value_t> result;
    result = cross_validate_trainer_threaded(trainer, samples, labels, this->trainer.getCvFolds(), this->trainer.getCvFolds());

    this->trainer.getLogbook()(Severity::Trace) << "CV SEN " << result(0) << " SPE " << result(1) << " at gamma = " << gamma << " and C = " << c;

    if (sum(result) > best_value || best_value == 0) {
      best_value = sum(result);
      best_params = params;
    }

    return sum(result);
  }

  matrix<value_t> params() const {
    return best_params;
  }

  value_t value() const {
    return best_value;
  }

private:
  const std::vector<sample_t>& samples;
  const std::vector<value_t>& labels;
  const Trainer& trainer;
  mutable value_t best_value;
  mutable matrix<value_t> best_params;
};

void create_fold(std::vector<float>& samples, std::vector<float>& labels, int iFold, int foldCount,
    std::vector<float>& trainingSamples, std::vector<float>& trainingLabels,
    std::vector<float>& testSamples, std::vector<float>& testLabels)
{
  const size_t sampleCount = labels.size();
  const size_t featureCount = samples.size() / sampleCount;
  const int samplesPerFold = sampleCount / foldCount;
  const int usedSamples = foldCount * samplesPerFold;
  const int trainingSampleCount = (foldCount - 1) * samplesPerFold;

  trainingSamples.clear();
  trainingLabels.clear();
  testSamples.clear();
  testLabels.clear();

  for (int j = 0; j < featureCount; ++j) {
    for (int i = 0; i < usedSamples; ++i) {
      if (i / samplesPerFold == iFold)
        testSamples.push_back(samples[i + j * sampleCount]);
      else
        trainingSamples.push_back(samples[i + j * sampleCount]);
    }
  }
  for (int i = 0; i < usedSamples; ++i) {
    if (i / samplesPerFold == iFold)
      testLabels.push_back(labels[i]);
    else
      trainingLabels.push_back(labels[i]);
  }
}

struct gpu_objective {

  gpu_objective(const std::vector<float>& samples,
      const std::vector<float>& labels,
      const Trainer& trainer)
  : samples(samples), labels(labels), trainer(trainer), best_value(0) { }

  value_t operator()(const matrix<value_t>& params) const {
    const value_t gamma = exp(params(0));
    const value_t c = exp(params(1));

    std::vector<float> trainingSamples, trainingLabels, testSamples, testLabels;

    float* alphas;
    gpusvm::Kernel_params kp;
    kp.kernel_type = "rbf";
    kp.gamma = gamma;

    for (int iFold = 0; iFold < trainer.getCvFolds(); ++iFold) {
      create_fold(samples, labels, iFold, trainer.getCvFolds(), trainingSamples, trainingLabels, testSamples, testLabels);

      gpusvm::performTraining(&trainingSamples[0], trainingLabels.size(),
          trainingSamples.size() / trainingLabels.size(), &trainingLabels[0],
          &alphas, &kp, c);

      // form the model
      // classify test set
      // add total pos, total neg, true pos, true negs.

      delete [] alphas;
    }

    // calculate SEN and SPE based on counts

    this->trainer.getLogbook()(Severity::Trace) << "CV SEN " << result(0) << " SPE " << result(1) << " at gamma = " << gamma << " and C = " << c;

    if (sum(result) > best_value || best_value == 0) {
      best_value = sum(result);
      best_params = params;
    }

    return sum(result);
  }

  matrix<value_t> params() const {
    return best_params;
  }

  value_t value() const {
    return best_value;
  }

private:
  const std::vector<float>& samples;
  const std::vector<float>& labels;
  const Trainer& trainer;
  mutable value_t best_value;
  mutable matrix<value_t> best_params;
};

void Trainer::update(gapputils::workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  std::vector<boost::shared_ptr<image_t> >& featureMaps = *getFeatureMaps();
  std::vector<boost::shared_ptr<image_t> >& segmentations = *getSegmentations();

  if (featureMaps.size() != segmentations.size()) {
    dlog(Severity::Warning) << "Maps and segmentations count does not match. Aborting!";
    return;
  }

  std::vector<sample_t> samples;
  std::vector<value_t> labels;

  std::vector<sample_t> cvsamples;
  std::vector<value_t> cvlabels;

  const size_t featureCount = featureMaps[0]->getSize()[2];
  const size_t sampleCount = featureMaps.size();

  for (size_t iSample = 0; iSample < featureMaps.size(); ++iSample) {

    image_t& features = *featureMaps[iSample];
    image_t& segmentation = *segmentations[iSample];

    if (features.getSize()[0] != segmentation.getSize()[0] ||
        features.getSize()[1] != segmentation.getSize()[1] ||
        segmentation.getSize()[2] != 1)
    {
      dlog(Severity::Warning) << "Dimensions do not match. Aborting!";
      return;
    }

    // fill samples and labels
    sample_t sample;
    sample.set_size(features.getSize()[2]);

    size_t slice = segmentation.getCount();

    float* data = features.getData();
    for (size_t i = 0; i < slice; ++i) {
      for (int j = 0; j < sample.size(); ++j) {
        sample(j) = data[i + j * slice];
      }
      samples.push_back(sample);
      if ((int)iSample < getCvImageCount())
        cvsamples.push_back(sample);
    }

    for (size_t i = 0; i < segmentation.getCount(); ++i) {
      labels.push_back(segmentation.getData()[i] > 0.5 ? 1 : -1);
      if ((int)iSample < getCvImageCount())
        cvlabels.push_back(segmentation.getData()[i] > 0.5 ? 1 : -1);
    }
  }
  dlog() << "Dataset collected.";

  // Normalize and randomize data
  vector_normalizer<sample_t> normalizer;
  normalizer.train(samples);
  std::transform(samples.begin(), samples.end(), samples.begin(), normalizer);
  std::transform(cvsamples.begin(), cvsamples.end(), cvsamples.begin(), normalizer);
  dlog() << "Dataset normalized.";

  if (getRandomizeSamples()) {
    randomize_samples(samples, labels);
    randomize_samples(cvsamples, cvlabels);
    dlog() << "Dataset randomized.";
  }

  std::vector<float> gpusamples;
  std::vector<float> gpulabels;

  for (size_t i = 0; i < featureCount; ++i) {
    for (size_t j = 0; j < sampleCount; ++j)
      gpusamples.push_back(samples[j](i));
  }

  for (size_t j = 0; j < sampleCount; ++j)
    gpulabels.push_back(labels[j]);

//  const value_t max_nu = 0.999 * maximum_nu(labels);

  matrix<value_t> params = cartesian_product(logspace(log10(getMaxGamma()), log10(getMin()), getSteps()),
      logspace(log10(getMaxC()), log10(getMin()), getSteps()));

  const int maxSteps = params.nc() + getMaxIterations() + 1;

  objective obj(
      (getCvImageCount() > 0 ? cvsamples : samples),
      (getCvImageCount() > 0 ? cvlabels : labels),
      *this);

  value_t best_result;
  best_result = 0;
  value_t best_gamma = 0.1, best_c = getMaxC();
  for (long col = 0; col < params.nc() && (monitor ? !monitor->getAbortRequested() : true); ++col) {
    value_t result = obj(log(colm(params, col)));
    if (result > best_result) {
      best_result = result;
      best_gamma = params(0, col);
      best_c = params(1, col);
    }
    if (monitor)
      monitor->reportProgress(100. * col / maxSteps);
  }
  dlog() << "Best result of grid search: " << best_result << " at gamma = " << best_gamma << " and C = " << best_c;

  // Optimization of parameters
  params.set_size(2,1);
  params = best_gamma, best_c;

  matrix<value_t> lower_bound(2,1), upper_bound(2,1);
  lower_bound = getMin() / 10., getMin() / 10.;
  upper_bound = 10. * getMaxGamma(), getMaxC();

  params = log(params);
  lower_bound = log(lower_bound);
  upper_bound = log(upper_bound);

  try {

    find_max_bobyqa(obj, params, params.size() * 2 + 1,
        lower_bound, upper_bound, min(upper_bound - lower_bound) / 10,
        getTolerance(), getMaxIterations()
    );

  } catch (std::exception& e) {
    dlog(Severity::Warning) << "Parameter search aborted: " << e.what();
  }

  params = exp(obj.params());
  dlog() << "Best result of BOBYQA: " << obj.value() << " at gamma = " << params(0) << " and C = " << params(1);

  svm_c_trainer<kernel_t> trainer;
  trainer.set_kernel(kernel_t(params(0)));
  trainer.set_c(params(1));

  typedef normalized_function<decision_function<kernel_t> > function_t;

  function_t f;
  f.normalizer = normalizer;
  if (getRank() > 0)
    f.function = reduced2(trainer, getRank()).train(samples, labels);
  else
    f.function = trainer.train(samples, labels);

  dlog() << "Number of support vectors: " << f.function.basis_vectors.nr();

  std::ofstream model(getModelName().c_str(), std::ios::binary);
  serialize(f, model);
  model.close();

  getHostInterface()->saveDataModel(getModelName() + ".xml");

  newState->setOutputName(getModelName());
}

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
