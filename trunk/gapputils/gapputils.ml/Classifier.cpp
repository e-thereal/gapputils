/*
 * Classifier.cpp
 *
 *  Created on: Oct 5, 2012
 *      Author: tombr
 */

#include "Classifier.h"

#include <fstream>

#include <dlib/svm.h>

#include <capputils/FilenameAttribute.h>
#include <capputils/FileExistsAttribute.h>
#include <boost/lambda/lambda.hpp>

#include <algorithm>

#include "SupportVectorClassifier.h"

namespace gapputils {
namespace ml {
namespace segmentation {

using namespace capputils::attributes;
using namespace gapputils::attributes;

BeginPropertyDefinitions(Classifier)

  ReflectableBase(workflow::DefaultWorkflowElement<Classifier>)

  WorkflowProperty(Features, Input("F"), NotNull<Type>())
  WorkflowProperty(ModelName, Filename(), FileExists())
  WorkflowProperty(Segmentation, Output("Seg"))

EndPropertyDefinitions

Classifier::Classifier() {
  setLabel("Classifier");
}

Classifier::~Classifier() { }

void Classifier::update(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace dlib;

  using namespace boost::lambda;

  typedef float value_t;
  typedef matrix<value_t, 0, 1> sample_t;
//  typedef radial_basis_kernel<sample_t> kernel_t;
//  typedef normalized_function<decision_function<kernel_t> > function_t;

  float* features = getFeatures()->getData();

  vector_normalizer<sample_t> normalizer;
  SupportVectorClassifier svc;

  std::ifstream model(getModelName().c_str(), std::ios::binary);
  deserialize(normalizer, model);
  svc.deserialize(model);
  model.close();

//  std::ifstream model(getModelName().c_str(), std::ios::binary);
//  function_t df;
//  deserialize(df, model);
//  model.close();

  image_t::dim_t size = {getFeatures()->getSize()[0], getFeatures()->getSize()[1], 1};
  image_t::dim_t pixelSize = {1000, 1000, 1000};

  boost::shared_ptr<image_t> segmentation(new image_t(size, pixelSize));
  size_t count = segmentation->getCount();
  float* result = segmentation->getData();

  const size_t dimensions = getFeatures()->getSize()[2];

  sample_t sample;
  sample.set_size(getFeatures()->getSize()[2]);

  std::vector<sample_t> samples;

  for (size_t i = 0; i < count; ++i) {
    for (int j = 0; j < sample.size(); ++j) {
      sample(j) = features[i + j * count];
    }
    samples.push_back(normalizer(sample));
//    result[i] = df(sample) > 0;
  }

  std::vector<float> gpusamples;
  std::vector<float> gpulabels;

  for (size_t i = 0; i < dimensions; ++i) {
    for (size_t j = 0; j < count; ++j)
      gpusamples.push_back(samples[j](i));
  }
  gpulabels.resize(count);
  svc.classify(gpusamples, gpulabels);

  std::transform(gpulabels.begin(), gpulabels.end(), result, _1 > 0.f);
  newState->setSegmentation(segmentation);
}

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
