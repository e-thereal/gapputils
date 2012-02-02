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

#include <gapputils/HideAttribute.h>
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
  DefineProperty(Image, Input(), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MinValue, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MaxValue, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(AutoScale, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MifName, Output("Mif"), Filename(), NotEqual<std::string>(""), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

ImageToMif::ImageToMif() : _MinValue(0), _MaxValue(1), _AutoScale(true), data(0) {
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

  const int columnCount = getImage()->getSize().x;
  const int rowCount = getImage()->getSize().y;
  const int sliceCount = getImage()->getSize().z;

  CMIF mif(columnCount, rowCount, sliceCount);
  mif.getChannel(1).setPixelSizeX(getImage()->getVoxelSize().x);
  mif.getChannel(1).setPixelSizeY(getImage()->getVoxelSize().y);
  mif.getChannel(1).setSliceThickness(getImage()->getVoxelSize().z);
  CMIF::pixelArray pixels = mif.getRawData();

  float* features = getImage()->getWorkingCopy();

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
        pixels[z][y][x] = std::min(2048.0, std::max(0.0, (features[i] - minV) * 2048. / (maxV - minV)));
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
