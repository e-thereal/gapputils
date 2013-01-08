/*
 * ImageViewer.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewer.h"

#include <capputils/TimeStampAttribute.h>

#include <capputils/EventHandler.h>

namespace gml {

namespace imaging {

namespace ui {

int ImageViewer::backgroundId;
int ImageViewer::modeId;

BeginPropertyDefinitions(ImageViewer)

  ReflectableBase(DefaultWorkflowElement<ImageViewer>)

  WorkflowProperty(BackgroundImage, Input(""), TimeStamp(backgroundId = Id))
  WorkflowProperty(Mode, Enumerator<Type>(), TimeStamp(modeId = Id))
  WorkflowProperty(WobbleDelay);

EndPropertyDefinitions

ImageViewer::ImageViewer() : _WobbleDelay(100) {
  setLabel("Viewer");
}

ImageViewer::~ImageViewer() {
  if (dialog) {
    dialog->close();
  }
}

void ImageViewer::show() {
  if (!dialog) {
    dialog = boost::make_shared<ImageViewerDialog>(this);
  }
  dialog->show();
}

}

}

}
