/*
 * ImageToMif.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#include "ImageToMif.h"

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
#include <gapputils/ReadOnlyAttribute.h>

#include <CMIF.hpp>
#include <CSlice.hpp>
#include <CProcessInfo.hpp>
#include <CChannel.hpp>
#include <iter/CPixelIterators.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(ImageToMif)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Image, Input(), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(MinValue, Observe(Id), TimeStamp(Id))
  DefineProperty(MaxValue, Observe(Id), TimeStamp(Id))
  DefineProperty(MaximumIntensity, Observe(Id), TimeStamp(Id))
  DefineProperty(AutoScale, Observe(Id), TimeStamp(Id))
  DefineProperty(MifName, Output("Mif"), Filename(), NotEqual<std::string>(""), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

ImageToMif::ImageToMif() : _MinValue(0), _MaxValue(1), _MaximumIntensity(2048), _AutoScale(true), data(0) {
  WfeUpdateTimestamp
  setLabel("ImageToMif");

  static char** argv = new char*[1];
  argv[0] = "FeaturesToMif";
  MSMRI::CProcessInfo::getInstance().getCommandLine(1, argv);

  Changed.connect(capputils::EventHandler<ImageToMif>(this, &ImageToMif::changedHandler));
}

ImageToMif::~ImageToMif() {
  if (data)
    delete data;
}

void ImageToMif::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ImageToMif::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;

  if (!data)
    data = new ImageToMif();

  if (!capputils::Verifier::Valid(*this) || !getImage())
    return;

  const int columnCount = getImage()->getSize()[0];
  const int rowCount = getImage()->getSize()[1];
  const int sliceCount = getImage()->getSize()[2];

  double maxIntens = getMaximumIntensity();

  CMIF mif(columnCount, rowCount, sliceCount);
  mif.getChannel(1).setPixelSizeX(getImage()->getPixelSize()[0]);
  mif.getChannel(1).setPixelSizeY(getImage()->getPixelSize()[1]);
  mif.getChannel(1).setSliceThickness(getImage()->getPixelSize()[2]);
  CMIF::pixelArray pixels = mif.getRawData();

  float* features = getImage()->getData();

  float minV = features[0], maxV = features[0];

  if (getAutoScale()) {
    for (int z = 1, i = 0; z <= mif.getSliceCount(); ++z) {
      for (int y = 0; y < mif.getRowCount(); ++y) {
        for (int x = 0; x < mif.getColumnCount(); ++x, ++i) {
          minV = std::min(minV, features[i]);
          maxV = std::max(maxV, features[i]);
        }
      }
      if (monitor)
        monitor->reportProgress(50 * z / mif.getSliceCount());
    }
  } else {
    minV = getMinValue();
    maxV = getMaxValue();
  }

  for (int z = 1, i = 0; z <= mif.getSliceCount(); ++z) {
    for (int y = 0; y < mif.getRowCount(); ++y) {
      for (int x = 0; x < mif.getColumnCount(); ++x, ++i) {
        pixels[z][y][x] = std::min(2048.0, std::max(0.0, (features[i] - minV) * maxIntens / (maxV - minV)));
      }
    }
    if (monitor) {
      if (getAutoScale()) {
        monitor->reportProgress(50 * z / mif.getSliceCount() + 50);
      } else {
        monitor->reportProgress(100 * z / mif.getSliceCount());
      }
    }
  }

  mif.writeToFile(getMifName(), true);
}

void ImageToMif::writeResults() {
  if (!data)
    return;

  setMifName(getMifName());
}

}

}
