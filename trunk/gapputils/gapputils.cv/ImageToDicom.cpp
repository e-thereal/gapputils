/*
 * ImageToDicom.cpp
 *
 *  Created on: Dec 08, 2011
 *      Author: tombr
 */

#include "ImageToDicom.h"

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
#include <gapputils/ReadOnlyAttribute.h>

#include <milib/ImageFactory.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageToDicom)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Image, Input("Img"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(MinValue, Observe(Id), TimeStamp(Id))
  DefineProperty(MaxValue, Observe(Id), TimeStamp(Id))
  DefineProperty(AutoScale, Observe(Id), TimeStamp(Id))
  DefineProperty(Filename, Output("Name"), Filename(), NotEqual<std::string>(""), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

ImageToDicom::ImageToDicom()
 : _MinValue(0), _MaxValue(1), _AutoScale(true), data(0)
{
  WfeUpdateTimestamp
  setLabel("ImageToDicom");

  Changed.connect(capputils::EventHandler<ImageToDicom>(this, &ImageToDicom::changedHandler));
}

ImageToDicom::~ImageToDicom() {
  if (data)
    delete data;
}

void ImageToDicom::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageToDicom::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ImageToDicom();

  if (!capputils::Verifier::Valid(*this) || !getImage())
    return;

  image_t& image = *getImage();
  const int width = image.getSize()[0];
  const int height = image.getSize()[1];
  const int depth = image.getSize()[2];

  const int count = width * height * depth;

  float* pixels = image.getData();
  //for (int i = 0; i < count; ++i)
  //  std::cout << pixels[i] << " ";

  float minV = pixels[0], maxV = pixels[0];

  if (getAutoScale()) {
    for (int i = 0; i < count; ++i) {
      minV = std::min(minV, pixels[i]);
      maxV = std::max(maxV, pixels[i]);
    }
  } else {
    minV = getMinValue();
    maxV = getMaxValue();
  }

  milib::IMedicalImage& dicom = milib::ImageFactory::GetInstance().CreateImage(width, height, depth);
  dicom.SetNormalizedBuffer(pixels, count, -minV, 255.f / maxV);
  dicom.SaveAs(getFilename());
}

void ImageToDicom::writeResults() {
  if (!data)
    return;

  setFilename(getFilename());
}

}

}
