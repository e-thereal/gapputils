/*
 * ImageSaver.cpp
 *
 *  Created on: Aug 15, 2011
 *      Author: tombr
 */

#include "ImageSaver.h"

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

#include <gapputils/HideAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageSaver)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ImagePtr, Input("Img"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ImageName, Output("Name"), Filename("Images (*.jpg, *.png)"), NotEqual<std::string>(""), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ImageSaver::ImageSaver() : data(0) {
  WfeUpdateTimestamp
  setLabel("ImageSaver");

  Changed.connect(capputils::EventHandler<ImageSaver>(this, &ImageSaver::changedHandler));
}

ImageSaver::~ImageSaver() {
  if (data)
    delete data;
}

void ImageSaver::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageSaver::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageSaver();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getImagePtr())
    return;

  getImagePtr()->save(getImageName().c_str());
}

void ImageSaver::writeResults() {
  if (!data)
    return;

}

}

}
