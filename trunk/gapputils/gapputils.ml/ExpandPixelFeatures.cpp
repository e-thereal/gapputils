/*
 * ExpandPixelFeatures.cpp
 *
 *  Created on: Oct 18, 2012
 *      Author: tombr
 */

#include "ExpandPixelFeatures.h"

#include <capputils/NotEqualAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {
namespace ml {

BeginPropertyDefinitions(ExpandPixelFeatures)

  ReflectableBase(workflow::DefaultWorkflowElement<ExpandPixelFeatures>)

  WorkflowProperty(Features, Input("Data"), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Images, Output("Imgs"))
  WorkflowProperty(Width, NotEqual<Type>(0))
  WorkflowProperty(Height, NotEqual<Type>(0))
  WorkflowProperty(FeatureCount, NotEqual<Type>(0))

EndPropertyDefinitions

ExpandPixelFeatures::ExpandPixelFeatures() : _Width(0), _Height(0), _FeatureCount(0) {
  setLabel("ExpandPFs");
}

ExpandPixelFeatures::~ExpandPixelFeatures() {
}

void ExpandPixelFeatures::update(workflow::IProgressMonitor* monitor) const {
  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > images(new std::vector<boost::shared_ptr<image_t> >());

  std::vector<float>& features = *getFeatures();
  const size_t width = getWidth();
  const size_t height = getHeight();
  const size_t pixelCount = width * height;
  const size_t featureCount = getFeatureCount();
  const size_t count = pixelCount * featureCount;

  for (size_t offset = 0; offset < features.size(); offset += count) {

    boost::shared_ptr<image_t> image(new image_t(width, height, featureCount));
    float* data = image->getData();
    for (size_t iFeature = 0; iFeature < featureCount; ++iFeature) {

      for (size_t iPixel = 0; iPixel < pixelCount; ++iPixel) {
        data[iPixel + iFeature * pixelCount] = features[offset + iPixel * featureCount + iFeature];
      }
    }
    images->push_back(image);
  }

  newState->setImages(images);
}

} /* namespace ml */
} /* namespace gapputils */
