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
#include <capputils/FilenameAttribute.h>
#include <capputils/EnumeratorAttribute.h>

//#include <capputils/Logbook.h>

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
  WorkflowProperty(Max)
  WorkflowProperty(Steps)
  WorkflowProperty(Tolerance)
  WorkflowProperty(MaxIterations)
  WorkflowProperty(CvFolds, Description("Number of folds used for k-fold cross validation."))
  WorkflowProperty(CvImageCount, Description("Number of images used to estimate hyperparameters using cross validation. A count of -1 denotes that all images should be used."))
  WorkflowProperty(RandomizeSamples)
  WorkflowProperty(OutputName, Output("SVC"), Filename(), Description("Will be set to ModelName after the update."))

EndPropertyDefinitions

Trainer::Trainer() : _Rank(10), _Min(1e-5), _Max(1), _Steps(4), _Tolerance(0.01), _MaxIterations(10),
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
    const value_t nu = exp(params(1));

    svm_nu_trainer<kernel_t> trainer;
    trainer.set_kernel(kernel_t(gamma));
    trainer.set_nu(nu);

    matrix<value_t> result;
    result = cross_validate_trainer_threaded(trainer, samples, labels, this->trainer.getCvFolds(), this->trainer.getCvFolds());

    this->trainer.getLogbook()(Severity::Trace) << "CV SEN " << result(0) << " SPE " << result(1) << " at gamma = " << gamma << " and nu = " << nu;

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

  const value_t max_nu = 0.999 * maximum_nu(labels);

  matrix<value_t> params = cartesian_product(logspace(log10(getMax()), log10(getMin()), getSteps()),
      logspace(log10(max_nu), log10(getMin()), getSteps()));

  const int maxSteps = params.nc() + getMaxIterations() + 1;

  objective obj(
      (getCvImageCount() > 0 ? cvsamples : samples),
      (getCvImageCount() > 0 ? cvlabels : labels),
      *this);

  value_t best_result;
  best_result = 0;
  value_t best_gamma = 0.1, best_nu = max_nu;
  for (long col = 0; col < params.nc() && (monitor ? !monitor->getAbortRequested() : true); ++col) {
    value_t result = obj(log(colm(params, col)));
    if (result > best_result) {
      best_result = result;
      best_gamma = params(0, col);
      best_nu = params(1, col);
    }
    if (monitor)
      monitor->reportProgress(100. * col / maxSteps);
  }
  dlog() << "Best result of grid search: " << best_result << " at gamma = " << best_gamma << " and nu = " << best_nu;

  // Optimization of parameters
  params.set_size(2,1);
  params = best_gamma, best_nu;

  matrix<value_t> lower_bound(2,1), upper_bound(2,1);
  lower_bound = getMin() / 10., getMin() / 10.;
  upper_bound = 10. * getMax(), max_nu;

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
  dlog() << "Best result of BOBYQA: " << obj.value() << " at gamma = " << params(0) << " and nu = " << params(1);

  svm_nu_trainer<kernel_t> trainer;
  trainer.set_kernel(kernel_t(params(0)));
  trainer.set_nu(params(1));

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
