/*
 * ImageViewer.cpp
 *
 *  Created on: Jul 13, 2011
 *      Author: tombr
 */

#include "ImageViewer.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/Xmlizer.h>

#include <capputils/NoParameterAttribute.h>
#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include "ImageViewerDialog.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int ImageViewer::backgroundId;

BeginPropertyDefinitions(ImageViewer)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(BackgroundImage, Input(""), Volatile(), ReadOnly(), Observe(backgroundId = Id), TimeStamp(Id))

EndPropertyDefinitions

ImageViewer::ImageViewer() : dialog(0)
{
  WfeUpdateTimestamp
  setLabel("Viewer");

  Changed.connect(capputils::EventHandler<ImageViewer>(this, &ImageViewer::changedHandler));
}

ImageViewer::~ImageViewer() {
  if (dialog) {
    dialog->close();
    delete dialog;
  }
}

void ImageViewer::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == backgroundId) {
    if (getBackgroundImage() && dialog) {
      QImage& bg = *getBackgroundImage();
      ImageViewerWidget* widget = (ImageViewerWidget*)dialog->getWidget();
      widget->updateSize(bg.width(), bg.height());
      widget->setBackgroundImage(getBackgroundImage());
    }
  }
}

void ImageViewer::execute(gapputils::workflow::IProgressMonitor* monitor) const {
}

void ImageViewer::writeResults() {
}

void ImageViewer::show() {
  if (!dialog) {
    ImageViewerWidget* widget = new ImageViewerWidget(100, 100);
    if (getBackgroundImage()) {
      widget->updateSize(getBackgroundImage()->width(), getBackgroundImage()->height());
      widget->setBackgroundImage(getBackgroundImage());
    }
    dialog = new ImageViewerDialog(widget);
  }
  dialog->show();
}

}

}
