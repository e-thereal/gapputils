/*
 * MakeImage.cpp
 *
 *  Created on: Jun 13, 2013
 *      Author: tombr
 */

#include "MakeImage.h"
#include <capputils/attributes/GreaterThanAttribute.h>

#include <algorithm>

namespace gml {

namespace imaging {

namespace core {

BeginPropertyDefinitions(MakeImage)

  ReflectableBase(DefaultWorkflowElement<MakeImage>)

  WorkflowProperty(Data, Input("D"))
  WorkflowProperty(Datas, Input("Ds"))
  WorkflowProperty(Width, GreaterThan<Type>(0))
  WorkflowProperty(Height, GreaterThan<Type>(0))
  WorkflowProperty(Depth, GreaterThan<Type>(0))
  WorkflowProperty(Image, Output("I"))
  WorkflowProperty(Images, Output("Is"))

EndPropertyDefinitions

MakeImage::MakeImage() : _Width(0), _Height(0), _Depth(0) {
  setLabel("D2I");
}

void MakeImage::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getData()) {
    data_t& data = *getData();

    boost::shared_ptr<image_t> image(new image_t(getWidth(), getHeight(), getDepth()));
    if (image->getCount() != data.size()) {
      dlog(Severity::Warning) << "Number of elements of the data vector does not match the given image size. Aborting!";
      return;
    }

    std::copy(data.begin(), data.end(), image->begin());
    newState->setImage(image);
  } else if (getDatas()) {
    v_data_t& datas = *getDatas();

    boost::shared_ptr<v_image_t> images(new v_image_t());
    for (size_t i = 0; i < datas.size(); ++i) {
      data_t& data = *datas[i];

      boost::shared_ptr<image_t> image(new image_t(getWidth(), getHeight(), getDepth()));
      if (image->getCount() != data.size()) {
        dlog(Severity::Warning) << "Number of elements of the data vector does not match the given image size. Skipping image " << i << "!";
        continue;
      }

      std::copy(data.begin(), data.end(), image->begin());
      images->push_back(image);
    }
    newState->setImages(images);
  }
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
