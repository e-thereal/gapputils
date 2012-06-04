/*
 * ImageToFeatures.cpp
 *
 *  Created on: Dec 08, 2011
 *      Author: tombr
 */

#include "ImageToFeatures.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/NoParameterAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <culib/CudaImage.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int ImageToFeatures::dataId;

BeginPropertyDefinitions(ImageToFeatures)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Image, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Data, Output(""), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Width, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Depth, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  

EndPropertyDefinitions

ImageToFeatures::ImageToFeatures()
 : _Width(0), _Height(0), _Depth(0), data(0)
{
  WfeUpdateTimestamp
  setLabel("I2F");

  Changed.connect(capputils::EventHandler<ImageToFeatures>(this, &ImageToFeatures::changedHandler));
}

ImageToFeatures::~ImageToFeatures() {
  if (data)
    delete data;
}

void ImageToFeatures::changedHandler(capputils::ObservableClass* sender, int eventId) {
  /*if (eventId == dataId) {
    execute(0);
    writeResults();
  }*/
}

void ImageToFeatures::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageToFeatures();

  if (!capputils::Verifier::Valid(*this) || !getImage())
    return;

  culib::ICudaImage& image = *getImage();

  const int width = image.getSize().x;
  const int height = image.getSize().y;
  const int depth = image.getSize().z;
  const int count = width * height * depth;
  
  //image.saveDeviceToWorkingCopy();

  boost::shared_ptr<std::vector<float> > output(new std::vector<float>(count));
  std::copy(image.getWorkingCopy(), image.getWorkingCopy() + count, output->begin());

  data->setWidth(width);
  data->setHeight(height);
  data->setDepth(depth);
  data->setData(output);
}

void ImageToFeatures::writeResults() {
  if (!data)
    return;

  setWidth(data->getWidth());
  setHeight(data->getHeight());
  setDepth(data->getDepth());
  setData(data->getData());
}

}

}

