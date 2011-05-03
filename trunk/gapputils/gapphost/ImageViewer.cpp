#include "ImageViewer.h"

#include <LabelAttribute.h>
#include <InputAttribute.h>
#include "ImageViewerItem.h"
#include "CustomToolItemAttribute.h"
#include <ObserveAttribute.h>
#include <HideAttribute.h>
#include <VolatileAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

using namespace attributes;

BeginPropertyDefinitions(ImageViewer, CustomToolItem<ImageViewerItem>())

DefineProperty(Label, Label(), Observe(PROPERTY_ID))
DefineProperty(ImagePtr, Input(), Observe(PROPERTY_ID), Hide(), Volatile())

EndPropertyDefinitions

ImageViewer::ImageViewer(void) : changeHandler(this), _Label("Viewer"), _ImagePtr(0)
{
  dialog = new ShowImageDialog();
  dialog->setWindowTitle(QString("Image Viewer: ") + getLabel().c_str());
  Changed.connect(changeHandler);
}

ImageViewer::~ImageViewer(void)
{
  delete dialog;
}

void ImageViewer::showImage() {
  QImage* image = getImagePtr();
  if (image)
    dialog->setImage(image);
  dialog->show();
}

}
