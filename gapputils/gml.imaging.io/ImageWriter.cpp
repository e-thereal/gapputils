/*
 * ImageWriter.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#include "ImageWriter.h"

#include <capputils/EventHandler.h>
#include <capputils/TimeStampAttribute.h>

#include <qimage.h>
#include <qcolor.h>

#include <sstream>
#include <iomanip>

namespace gml {

namespace imaging {

namespace io {

int ImageWriter::imageId;

BeginPropertyDefinitions(ImageWriter)
  ReflectableBase(DefaultWorkflowElement<ImageWriter>)

  WorkflowProperty(Image, Input("Img"), NotNull<Type>(), TimeStamp(imageId = Id))
  WorkflowProperty(Filename, Filename("Images (*.jpg *.png)"), NotEmpty<Type>())
  WorkflowProperty(AutoSave)
  WorkflowProperty(AutoName)
  WorkflowProperty(AutoSuffix)
  WorkflowProperty(OutputName, Output("Name"))
EndPropertyDefinitions

ImageWriter::ImageWriter() : _AutoSave(false), imageNumber(0) {
  setLabel("Writer");

  Changed.connect(capputils::EventHandler<ImageWriter>(this, &ImageWriter::changedHandler));
}

#define F_TO_INT(value) std::min(255, std::max(0, (int)(value * 256)))

void saveImage(image_t& image, const std::string& filename) {
  const int width = image.getSize()[0];
  const int height = image.getSize()[1];
  const int depth = image.getSize()[2];

  const int count = width * height;
  QImage qimage(width, height, QImage::Format_ARGB32);

  float* buffer = image.getData();

  for (int i = 0, y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      int r = F_TO_INT(buffer[i]);
      int g = depth == 3 ? F_TO_INT(buffer[i + count]) : r;
      int b = depth == 3 ? F_TO_INT(buffer[i + 2 * count]) : r;
      qimage.setPixel(x, y, QColor(r, g, b).rgb());
    }
  }

  qimage.save(filename.c_str());
}

void ImageWriter::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == imageId && getAutoSave() && getImage()) {
    std::stringstream filename;
    filename << getAutoName() << std::setw(8) << std::setfill('0') << imageNumber++ << getAutoSuffix();
    saveImage(*getImage(), filename.str());
  }
}

void ImageWriter::update(gapputils::workflow::IProgressMonitor* /*monitor*/) const {
  saveImage(*getImage(), getFilename());
  newState->setOutputName(getFilename());
}

}

}

}
