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
  WorkflowProperty(MinC)
  WorkflowProperty(MaxC)
  WorkflowProperty(CStep)

EndPropertyDefinitions

Trainer::Trainer() : _MinC(1e-3), _MaxC(1e5), _CStep(5) {
  setLabel("Trainer");
}

Trainer::~Trainer() { }

void Trainer::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace dlib;
  using namespace capputils;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  std::vector<boost::shared_ptr<image_t> >& featureMaps = *getFeatureMaps();
  std::vector<boost::shared_ptr<image_t> >& segmentations = *getSegmentations();

  if (featureMaps.size() != segmentations.size()) {
    dlog(Severity::Warning) << "Maps and segmentations count does not match. Aborting!";
    return;
  }

  typedef matrix<float, 0, 1> sample_t;

  std::vector<sample_t> samples;
  std::vector<double> labels;

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
    }

    for (size_t i = 0; i < segmentation.getCount(); ++i)
      labels.push_back(segmentation.getData()[i] > 0.5 ? 1 : -1);
  }

  // Normalize and randomize data
  vector_normalizer<sample_t> normalizer;
  normalizer.train(samples);
  std::transform(samples.begin(), samples.end(), samples.begin(), normalizer);

  randomize_samples(samples, labels);

  // perform cross validation to select C
  typedef linear_kernel<sample_t> kernel_t;
  svm_c_linear_trainer<kernel_t> trainer;

  double c = getMinC(), cend = getMaxC(), cbest = c, best = 0, current;
  int steps = log(cend / c) / log(getCStep());
  for (int i = 0; i < steps; ++i, c *= getCStep()) {
    trainer.set_c(c);

    current = sum(cross_validate_trainer(trainer, samples, labels, 4));
    if (current > best) {
      best = current;
      cbest = c;
    }
    dlog() << "Current cv error " << current << " at c = " << c;
    if (monitor)
      monitor->reportProgress(100.0 * (double)i / steps);
  }
  dlog() << "Best cv error " << best << " at c = " << cbest;
  trainer.set_c(cbest);

  decision_function<kernel_t> df = trainer.train(samples, labels);

  std::ofstream model(getModelName().c_str(), std::ios::binary);
  serialize(normalizer, model);
  serialize(df, model);
  model.close();
}

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
