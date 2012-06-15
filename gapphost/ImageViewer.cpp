#include "ImageViewer.h"

#include <gapputils/LabelAttribute.h>
#include <capputils/InputAttribute.h>
#include "ImageViewerItem.h"
#include "CustomToolItemAttribute.h"
#include <capputils/ObserveAttribute.h>
#include <gapputils/HideAttribute.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/EventHandler.h>

#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

using namespace attributes;

int ImageViewer::imageId;

BeginPropertyDefinitions(ImageViewer)

  ReflectableBase(workflow::WorkflowElement)
  DefineProperty(ImagePtr, Input("Img"), Observe(imageId = PROPERTY_ID), Hide(), Volatile())

EndPropertyDefinitions

ImageViewer::ImageViewer(void)
{
  setLabel("Viewer");

  dialog = new ShowImageDialog();
  dialog->setWindowTitle(QString("Image Viewer: ") + getLabel().c_str());
  Changed.connect(capputils::EventHandler<ImageViewer>(this, &ImageViewer::changeHandler));
}

ImageViewer::~ImageViewer(void)
{
  delete dialog;
}

void ImageViewer::execute(workflow::IProgressMonitor* monitor) const {
}

void ImageViewer::writeResults() { }

void ImageViewer::changeHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == labelId)
    dialog->setWindowTitle(QString("Image Viewer: ") + getLabel().c_str());
   
  if (eventId == imageId) {
    if (getImagePtr()) {
      dialog->setImage(getImagePtr().get());
    }
  }
}

void ImageViewer::show() {
  if (getImagePtr())
    dialog->setImage(getImagePtr().get());
  dialog->show();
}

}
