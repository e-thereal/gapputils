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

EndPropertyDefinitions

FlattenImage::FlattenImage() {
  setLabel("I2D");
}

void FlattenImage::update(IProgressMonitor* monitor) const {
  image_t& image = *getImage();

  auto output = boost::make_shared<std::vector<double> >(image.getCount());
  std::copy(image.begin(), image.end(), output->begin());
  newState->setData(output);
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
