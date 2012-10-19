/*
 * GroupPixelFeatures.cpp
 *
 *  Created on: Oct 18, 2012
 *      Author: tombr
 */

#include "GroupPixelFeatures.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(GroupPixelFeatures)

  ReflectableBase(workflow::DefaultWorkflowElement<GroupPixelFeatures>)

  WorkflowProperty(Images, Input("Imgs"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Features, Output("Data"));
  WorkflowProperty(PixelCount, NoParameter())
  WorkflowProperty(FeatureCount, NoParameter())
  WorkflowProperty(SampleCount, NoParameter())

EndPropertyDefinitions

GroupPixelFeatures::GroupPixelFeatures() : _PixelCount(0), _FeatureCount(0), _SampleCount(0) {
  setLabel("GroupPFs");
}

GroupPixelFeatures::~GroupPixelFeatures() {
}

void GroupPixelFeatures::update(workflow::IProgressMonitor* monitor) const {
  using capputils::Severity;

  boost::shared_ptr<std::vector<float> > features(new std::vector<float>());

  capputils::Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  std::vector<boost::shared_ptr<image_t> >& images = *getImages();

  const size_t pixelCount = images[0]->getSize()[0] * images[0]->getSize()[1];
  const size_t featureCount = images[0]->getSize()[2];

  for (size_t iImage = 0; iImage < images.size(); ++iImage) {
    image_t::value_t* data = images[iImage]->getData();

    for (size_t i = 0; i < pixelCount; ++i) {
      for (size_t j = 0; j < featureCount; ++j) {
        features->push_back(data[j * pixelCount + i]);
      }
    }
  }

  newState->setFeatures(features);
  newState->setFeatureCount(featureCount);
  newState->setPixelCount(pixelCount);
  newState->setSampleCount(pixelCount * images.size());
}

} /* namespace ml */
} /* namespace gapputils */
