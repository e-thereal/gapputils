/*
 * ImageDuplicator.cpp
 *
 *  Created on: Jan 29, 2014
 *      Author: tombr
 */

#include "ImageDuplicator.h"



namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(ImageDuplicator)

  ReflectableBase(DefaultWorkflowElement<ImageDuplicator>)

  WorkflowProperty(Image, Input(""), NotNull<Type>())
  WorkflowProperty(Count)
  WorkflowProperty(Outputs, Output(""))

EndPropertyDefinitions

ImageDuplicator::ImageDuplicator() : _Count(1) {
  setLabel("Copy");
}

void ImageDuplicator::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getCount() < 1) {
    dlog(Severity::Warning) << "Count must be greater than zero. Aborting!";
    return;
  }

  boost::shared_ptr<std::vector<boost::shared_ptr<image_t> > > outputs(new std::vector<boost::shared_ptr<image_t> >());
  for (int i = 0; i < getCount(); ++i)
    outputs->push_back(boost::make_shared<image_t>(*getImage()));

  newState->setOutputs(outputs);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
