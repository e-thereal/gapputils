#include "ImageReader.h"

#include <capputils/attributes/RenamedAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

#include <qimage.h>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(ImageReader, Renamed("gml::imaging::io::OpenImage"), Deprecated("Use OpenImage instead."))

  ReflectableBase(DefaultWorkflowElement<ImageReader>)

  WorkflowProperty(ImageName, Input("Name"), Filename("Images (*.jpg *.png)"), FileExists())
  WorkflowProperty(ImagePtr, Output("Img"))
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())

EndPropertyDefinitions

ImageReader::ImageReader() : _Width(0), _Height(0) {
  setLabel("Image");
}

void ImageReader::update(workflow::IProgressMonitor* monitor) const {
  Logbook& dlog = getLogbook();

  QImage qimage;
  if (!qimage.load(getImageName().c_str())) {
    dlog(Severity::Warning) << "Could not load image " << getImageName();
    return;
  }

  const int width = qimage.width();
  const int height = qimage.height();
  const int count = width * height;
  boost::shared_ptr<image_t> image(new image_t(width, height, 3));

  float* buffer = image->getData();
  for (int i = 0, y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      QRgb color = qimage.pixel(x, y);
      buffer[i] = (float)qRed(color) / 256.0;
      buffer[i + count] = (float)qGreen(color) / 256.0;
      buffer[i + 2 * count] = (float)qBlue(color) / 256.0;
    }
    if (monitor)
      monitor->reportProgress(100.0 * y / height);
  }

  newState->setWidth(width);
  newState->setHeight(height);
  newState->setImagePtr(image);
}

}

}

}
