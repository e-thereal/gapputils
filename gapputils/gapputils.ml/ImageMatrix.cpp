/*
 * ImageMatrix.cpp
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#include "ImageMatrix.h"

#include <capputils/DescriptionAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/Serializer.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <cmath>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int ImageMatrix::inputId;

BeginPropertyDefinitions(ImageMatrix)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("In"), Volatile(), ReadOnly(), Observe(inputId = Id), TimeStamp(Id))
  DefineProperty(MinValue, Observe(Id), TimeStamp(Id))
  DefineProperty(MaxValue, Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id),
      Description("The number of columns. A value of -1 indicates to always use a squared matrix."))
  DefineProperty(ImageMatrix, Output("Out"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(AutoScale, Observe(Id))
  DefineProperty(CenterImages, Observe(Id))

EndPropertyDefinitions

ImageMatrix::ImageMatrix() : _MinValue(-2), _MaxValue(2), _ColumnCount(-1), _AutoScale(false),
 _CenterImages(false), data(0)
{
  WfeUpdateTimestamp
  setLabel("Matrix");

  Changed.connect(capputils::EventHandler<ImageMatrix>(this, &ImageMatrix::changedHandler));
}

ImageMatrix::~ImageMatrix() {
  if (data)
    delete data;
}

void ImageMatrix::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId) {
    execute(0);
    writeResults();
  }
}

#define LOCATE(a,b) std::cout << #b": " << (char*)&a._##b - (char*)&a << std::endl
#define LOCATE2(a,b) std::cout << #b": " << (char*)&a.b - (char*)&a << std::endl

void ImageMatrix::writeResults() {
  if (!data)
    return;

  setImageMatrix(data->getImageMatrix());
}

}

}
