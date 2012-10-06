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

  typedef matrix<float, 0, 1> sample_t;
  typedef linear_kernel<sample_t> kernel_t;

  float* features = getFeatures()->getData();

  std::ifstream model(getModelName().c_str(), std::ios::binary);

  vector_normalizer<sample_t> normalizer;
  deserialize(normalizer, model);

  decision_function<kernel_t> df;
  deserialize(df, model);

  model.close();

  image_t::dim_t size = {getFeatures()->getSize()[0], getFeatures()->getSize()[1], 1};
  image_t::dim_t pixelSize = {1000, 1000, 1000};

  boost::shared_ptr<image_t> segmentation(new image_t(size, pixelSize));
  size_t count = segmentation->getCount();
  float* result = segmentation->getData();

  sample_t sample;
  sample.set_size(getFeatures()->getSize()[2]);

  for (size_t i = 0; i < count; ++i) {
    for (int j = 0; j < sample.size(); ++j) {
      sample(j) = features[i + j * count];
    }
    result[i] = df(normalizer(sample)) > 0;
  }

  newState->setSegmentation(segmentation);
}

} /* namespace segmentation */
} /* namespace ml */
} /* namespace gapputils */
