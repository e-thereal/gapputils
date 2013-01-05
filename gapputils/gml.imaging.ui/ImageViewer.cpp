/*
 * ImageViewer.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewer.h"

#include <capputils/EventHandler.h>

namespace gml {

namespace imaging {

namespace ui {

int ImageViewer::backgroundId;

BeginPropertyDefinitions(ImageViewer)

  ReflectableBase(DefaultWorkflowElement<ImageViewer>)

  DefineProperty(BackgroundImage, Input(""), Volatile(), ReadOnly(), Observe(backgroundId = Id))

EndPropertyDefinitions

ImageViewer::ImageViewer() {
  setLabel("Viewer");

  Changed.connect(capputils::EventHandler<ImageViewer>(this, &ImageViewer::changedHandler));
}

ImageViewer::~ImageViewer() {
  if (dialog) {
    dialog->close();
  }
}

void ImageViewer::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == backgroundId) {
    if (getBackgroundImage() && dialog) {
      dialog->setBackgroundImage(getBackgroundImage());
    }
  }
}

void ImageViewer::show() {
  if (!dialog) {
    dialog = boost::make_shared<ImageViewerDialog>();
    if (getBackgroundImage())
      dialog->setBackgroundImage(getBackgroundImage());
  }
  dialog->show();
}

}

}

}
