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

#include <capputils/HideAttribute.h>

#include <sstream>
#include <iomanip>

using namespace capputils::attributes;

namespace gapputils {

namespace cv {

int ImageSaver::imageId;

BeginPropertyDefinitions(ImageSaver)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(ImagePtr, Input("Img"), Hide(), Volatile(), Observe(imageId = Id), TimeStamp(Id))
  DefineProperty(ImageName, Output("Name"), Filename("Images (*.jpg *.png)"), NotEqual<std::string>(""), Observe(Id), TimeStamp(Id))
  DefineProperty(AutoSave, Observe(Id))
  DefineProperty(AutoName, Observe(Id))
  DefineProperty(AutoSuffix, Observe(Id))

EndPropertyDefinitions

ImageSaver::ImageSaver() : _AutoSave(false), data(0), imageNumber(0) {
  WfeUpdateTimestamp
  setLabel("ImageSaver");

  Changed.connect(capputils::EventHandler<ImageSaver>(this, &ImageSaver::changedHandler));
}

ImageSaver::~ImageSaver() {
  if (data)
    delete data;
}

void ImageSaver::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == imageId && getAutoSave() && getImagePtr()) {
    std::stringstream filename;
    filename << getAutoName() << std::setw(8) << std::setfill('0') << imageNumber++ << getAutoSuffix();
    getImagePtr()->save(filename.str().c_str());
  }
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
