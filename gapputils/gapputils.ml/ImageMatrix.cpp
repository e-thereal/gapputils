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

BeginPropertyDefinitions(ImageMatrix)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input("In"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MinValue, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MaxValue, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ImageMatrix, Output("Out"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ImageMatrix::ImageMatrix() : _MinValue(-2.f), _MaxValue(2.f), data(0) {
  WfeUpdateTimestamp
  setLabel("ImageMatrix");

  Changed.connect(capputils::EventHandler<ImageMatrix>(this, &ImageMatrix::changedHandler));
}

ImageMatrix::~ImageMatrix() {
  if (data)
    delete data;
}

void ImageMatrix::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageMatrix::writeResults() {
  if (!data)
    return;

  setImageMatrix(data->getImageMatrix());
}

}

}
