/*
 * ImageMatrix.cpp
 *
 *  Created on: Dec 17, 2011
 *      Author: tombr
 */

#include "ImageMatrix.h"

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

#include <culib/CudaImage.h>

#include <cmath>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

int ImageMatrix::inputId;

BeginPropertyDefinitions(ImageMatrix)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("In"), Volatile(), ReadOnly(), Observe(inputId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MinValue, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MaxValue, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ImageMatrix, Output("Out"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(AutoScale, Observe(PROPERTY_ID))
  DefineProperty(CenterImages, Observe(PROPERTY_ID))

EndPropertyDefinitions

ImageMatrix::ImageMatrix() : _MinValue(-2.f), _MaxValue(2.f), _AutoScale(false),
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
