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

int ImageViewer::imageId;
int ImageViewer::imagesId;
int ImageViewer::tensorId;
int ImageViewer::tensorsId;
int ImageViewer::modeId;
int ImageViewer::currentImageId;
int ImageViewer::currentSliceId;
int ImageViewer::minimumIntensityId;
int ImageViewer::maximumIntensityId;
int ImageViewer::minimumLengthId;
int ImageViewer::maximumLengthId;

BeginPropertyDefinitions(ImageViewer)

  ReflectableBase(DefaultWorkflowElement<ImageViewer>)

  WorkflowProperty(Image, Input("I"), Dummy(imageId = Id))
  WorkflowProperty(Images, Input("Is"), Dummy(imagesId = Id))
  WorkflowProperty(Tensor, Input("T"), Dummy(tensorId = Id))
  WorkflowProperty(Tensors, Input("Ts"), Dummy(tensorsId = Id))
  WorkflowProperty(AutoUpdateCurrentModule, Flag(), Description("If checked, the current module is updated on increment or decrement."))
  WorkflowProperty(AutoUpdateWorkflow, Flag(), Description("If checked, the current workflow is updated on increment or decrement."))
  WorkflowProperty(CurrentImage, Dummy(currentImageId = Id))
  WorkflowProperty(CurrentSlice, Dummy(currentSliceId = Id))
  WorkflowProperty(MinimumIntensity, Dummy(minimumIntensityId = Id))
  WorkflowProperty(MaximumIntensity, Dummy(maximumIntensityId = Id))
  WorkflowProperty(Contrast)
  WorkflowProperty(Mode, Enumerator<Type>(), Dummy(modeId = Id))
  WorkflowProperty(MinimumLength, Dummy(minimumLengthId = Id))
  WorkflowProperty(MaximumLength, Dummy(maximumLengthId = Id))
  WorkflowProperty(VisibleLength)
  //WorkflowProperty(WobbleDelay);

EndPropertyDefinitions

ImageViewer::ImageViewer()
 : _AutoUpdateCurrentModule(true), _AutoUpdateWorkflow(false), _CurrentImage(0), _CurrentSlice(0),
   _MinimumIntensity(0.0), _MaximumIntensity(1.0), _Contrast(1.0),
   _MinimumLength(0.0), _MaximumLength(1.0), _VisibleLength(1.0)//, _WobbleDelay(100)
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
