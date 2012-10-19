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
  WorkflowProperty(PixelCount, NotEqual<Type>(0))
  WorkflowProperty(FeatureCount, NotEqual<Type>(0))

EndPropertyDefinitions

ExpandPixelFeatures::ExpandPixelFeatures() : _PixelCount(0), _FeatureCount(0) {
  setLabel("ExpandPFs");
}

ExpandPixelFeatures::~ExpandPixelFeatures() {
}

void ExpandPixelFeatures::update(workflow::IProgressMonitor* monitor) const {
  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > images(new std::vector<boost::shared_ptr<image_t> >());



  newState->setImages(images);
}

} /* namespace ml */
} /* namespace gapputils */
