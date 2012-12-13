/*
 * ImageMatrix.cpp
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#include "ImageMatrix.h"

#include <capputils/EventHandler.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>

#include <cmath>

namespace gapputils {

namespace ml {

int ImageMatrix::inputId;

BeginPropertyDefinitions(ImageMatrix)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  WorkflowProperty(InputImage, Input("In"), NotNull<Type>(), TimeStamp(inputId = Id))
  WorkflowProperty(MinValue)
  WorkflowProperty(MaxValue)
  WorkflowProperty(ColumnCount,
      Description("The number of columns. A value of -1 indicates to always use a squared matrix."))
  WorkflowProperty(ImageMatrix, Output("Out"))
  WorkflowProperty(AutoScale)
  WorkflowProperty(CenterImages)
  WorkflowProperty(CroppedWidth)
  WorkflowProperty(CroppedHeight)

EndPropertyDefinitions

ImageMatrix::ImageMatrix() : _MinValue(-2), _MaxValue(2), _ColumnCount(-1), _AutoScale(false),
 _CenterImages(false), _CroppedWidth(-1), _CroppedHeight(-1)
{
  setLabel("Matrix");
  Changed.connect(capputils::EventHandler<ImageMatrix>(this, &ImageMatrix::changedHandler));
}

ImageMatrix::~ImageMatrix() { }

void ImageMatrix::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && Verifier::Valid(*this)) {
    execute(0);
    writeResults();
  }
}

ImageMatrixChecker imageMatrixChecker;

}

}
