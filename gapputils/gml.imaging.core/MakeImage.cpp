/*
 * MakeImage.cpp
 *
 *  Created on: Jun 13, 2013
 *      Author: tombr
 */

#include "MakeImage.h"

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(MakeImage)

  ReflectableBase(DefaultWorkflowElement<MakeImage>)

  WorkflowProperty(Data, Input(""), NotNull<Type>(), NotEmpty<Type>())
  WorkflowProperty(Width)
  WorkflowProperty(Height)
  WorkflowProperty(Depth)
  WorkflowProperty(Image, Output(""))

EndPropertyDefinitions

MakeImage::MakeImage() : _Width(0), _Height(0), _Depth(0) {
  setLabel("D2I");
}

void MakeImage::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  std::vector<double>& data = *getData();

  boost::shared_ptr<image_t> image(new image_t(getWidth(), getHeight(), getDepth()));
  if (image->getCount() != data.size()) {
    dlog(Severity::Warning) << "Number of elements of the data vector does not match the given image size. Aborting!";
    return;
  }

  std::copy(data.begin(), data.end(), image->begin());
  newState->setImage(image);
}

} /* namespace core */
} /* namespace imaging */
} /* namespace gml */
