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

int ImageViewer::imageId;
int ImageViewer::imagesId;
int ImageViewer::modeId;
int ImageViewer::currentImageId;
int ImageViewer::currentSliceId;
int ImageViewer::minimumIntensityId;
int ImageViewer::maximumIntensityId;

BeginPropertyDefinitions(ImageViewer)

  ReflectableBase(DefaultWorkflowElement<ImageViewer>)

  WorkflowProperty(Image, Input("I"), TimeStamp(imageId = Id))
  WorkflowProperty(Images, Input("Is"), TimeStamp(imagesId = Id))
  WorkflowProperty(CurrentImage, TimeStamp(currentImageId = Id))
  WorkflowProperty(CurrentSlice, TimeStamp(currentSliceId = Id))
  WorkflowProperty(MinimumIntensity, TimeStamp(minimumIntensityId = Id))
  WorkflowProperty(MaximumIntensity, TimeStamp(maximumIntensityId = Id))
  WorkflowProperty(Contrast)
  WorkflowProperty(Mode, Enumerator<Type>(), TimeStamp(modeId = Id))
  //WorkflowProperty(WobbleDelay);

EndPropertyDefinitions

ImageViewer::ImageViewer()
 : _CurrentImage(0), _CurrentSlice(0), _MinimumIntensity(0.0), _MaximumIntensity(1.0), _Contrast(1.0)//, _WobbleDelay(100)
{
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
