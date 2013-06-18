/*
 * FlattenImage.cpp
 *
 *  Created on: 2013-05-20
 *      Author: tombr
 */

#include "FlattenImage.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(FlattenImage)

  ReflectableBase(DefaultWorkflowElement<FlattenImage>)

  WorkflowProperty(Image, Input(""), NotNull<Type>())
  WorkflowProperty(Data, Output(""))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())

EndPropertyDefinitions

FlattenImage::FlattenImage() : _Width(0), _Height(0), _Depth(0) {
  setLabel("I2D");
}

void FlattenImage::update(IProgressMonitor* monitor) const {
  image_t& image = *getImage();

  boost::shared_ptr<std::vector<double> > output = boost::make_shared<std::vector<double> >(image.getCount());
  std::copy(image.begin(), image.end(), output->begin());
  newState->setData(output);
  newState->setWidth(image.getSize()[0]);
  newState->setHeight(image.getSize()[1]);
  newState->setDepth(image.getSize()[2]);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
