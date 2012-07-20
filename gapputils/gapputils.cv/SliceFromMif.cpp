/*
 * SliceFromMif.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "SliceFromMif.h"

#include <capputils/EnumeratorAttribute.h>
#include <capputils/DescriptionAttribute.h>
#include <capputils/NoParameterAttribute.h>
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
#include <CChannel.hpp>
#include <cmath>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(SliceFromMif)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(MifName, Input("Mif"), FileExists(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Image, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Width, Description("Is set to width of extracted slice."), NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Height, Description("Is set to height of extracted slice."), NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SlicePosition, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Orientation, Enumerator<SliceOrientation>(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

SliceFromMif::SliceFromMif() : _Width(0), _Height(0), _SlicePosition(0), data(0) {
  setLabel("SliceFromMif");

  Changed.connect(capputils::EventHandler<SliceFromMif>(this, &SliceFromMif::changedHandler));
}

SliceFromMif::~SliceFromMif() {
  if (data)
    delete data;
}

void SliceFromMif::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void SliceFromMif::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;

  if (!data)
    data = new SliceFromMif();

  if (!capputils::Verifier::Valid(*this))
    return;

  CMIF mif(getMifName());

  int width, height, slicePos;

  switch(getOrientation()) {
  case SliceOrientation::Axial:
    width = mif.getColumnCount();
    height = mif.getRowCount();
    slicePos = max(0, min(getSlicePosition(), mif.getSliceCount() - 1));
    break;
  case SliceOrientation::Sagital:
    width = mif.getRowCount();
    height = mif.getSliceCount();
    slicePos = max(0, min(getSlicePosition(), mif.getColumnCount() - 1));
    break;
  case SliceOrientation::Coronal:
    width = mif.getColumnCount();
    height = mif.getSliceCount();
    slicePos = max(0, min(getSlicePosition(), mif.getRowCount() - 1));
    break;
  }

  boost::shared_ptr<image_t> image(new image_t(width, height, 1,
      mif.getChannel(1).getPixelSizeX(),
      mif.getChannel(1).getPixelSizeY(),
      mif.getChannel(1).getSliceThickness()));
  float* buffer = image->getData();

  CMIF::pixelArray pixels = mif.getRawData();
  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      switch (getOrientation()) {
      case SliceOrientation::Axial:
        buffer[i] = (float)pixels[slicePos + 1][y][x] / 2048.f;
        break;
      case SliceOrientation::Sagital:
        buffer[i] = (float)pixels[mif.getSliceCount() - y][x][slicePos] / 2048.f;
        break;
      case SliceOrientation::Coronal:
        buffer[i] = (float)pixels[mif.getSliceCount() - y][slicePos][x] / 2048.f;
        break;
      }

    }
  }

  data->setWidth(width);
  data->setHeight(height);
  data->setImage(image);
}

void SliceFromMif::writeResults() {
  if (!data)
    return;

  setWidth(data->getWidth());
  setHeight(data->getHeight());
  setImage(data->getImage());
}

}

}
