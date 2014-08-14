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

  WorkflowProperty(Image, Input("I"))
  WorkflowProperty(Images, Input("Is"))
  WorkflowProperty(Data, Output("D"))
  WorkflowProperty(Datas, Output("Ds"))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())

EndPropertyDefinitions

FlattenImage::FlattenImage() : _Width(0), _Height(0), _Depth(0) {
  setLabel("I2D");
}

void FlattenImage::update(IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  if (getImage()) {
    image_t& image = *getImage();

    boost::shared_ptr<std::vector<double> > output = boost::make_shared<std::vector<double> >(image.getCount());
    std::copy(image.begin(), image.end(), output->begin());
    newState->setData(output);
    newState->setWidth(image.getSize()[0]);
    newState->setHeight(image.getSize()[1]);
    newState->setDepth(image.getSize()[2]);
  } else if (getImages() && getImages()->size() && getImages()->at(0)) {
    v_image_t& images = *getImages();

    const size_t width = images[0]->getSize()[0];
    const size_t height = images[0]->getSize()[1];
    const size_t depth = images[0]->getSize()[2];

    boost::shared_ptr<v_data_t> outputs(new v_data_t());

    for (size_t i = 0; i < images.size(); ++i) {
      image_t& image = *images[i];

      if (width != image.getSize()[0] || height != image.getSize()[1] || depth != image.getSize()[2]) {
        dlog(Severity::Warning) << "Image size mismatch (" <<  image.getSize()[0] << "x" << image.getSize()[1] << "x" << image.getSize()[2] << "). Skipping image " << i << "!";
        continue;
      }

      boost::shared_ptr<data_t> output = boost::make_shared<data_t>(image.getCount());
      std::copy(image.begin(), image.end(), output->begin());
      outputs->push_back(output);
    }

    newState->setDatas(outputs);
    newState->setWidth(width);
    newState->setHeight(height);
    newState->setDepth(depth);
  }
}

} /* namespace core */

} /* namespace imaging */

} /* namespace gml */
