/*
 * SliceFromMif.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "SliceFromMif.h"

#include <CMIF.hpp>
#include <CChannel.hpp>
#include <cmath>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(SliceFromMif)

  ReflectableBase(DefaultWorkflowElement<SliceFromMif>)

  WorkflowProperty(MifName, Input("Mif"), Filename("MIFs (*.MIF)"), FileExists())
  WorkflowProperty(Image, Output("Img"))
  WorkflowProperty(SlicePosition)
  WorkflowProperty(UseNormalizedIndex)
  WorkflowProperty(Orientation, Enumerator<Type>())
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(Width, Description("Is set to width of extracted slice."), NoParameter())
  WorkflowProperty(Height, Description("Is set to height of extracted slice."), NoParameter())

EndPropertyDefinitions

SliceFromMif::SliceFromMif() : _SlicePosition(0), _UseNormalizedIndex(false), _MaximumIntensity(2048),
  _Width(0), _Height(0)
{
  setLabel("SliceFromMif");
}

void SliceFromMif::update(gapputils::workflow::IProgressMonitor* /*monitor*/) const {
  using namespace MSMRI::MIF;
  using namespace std;

  CMIF mif(getMifName());

  int width = 0, height = 0, slicePos = 0;

  size_t pixelWidth = 1000, pixelHeight = 1000, pixelDepth = 1000;

  switch(getOrientation()) {
  case SliceOrientation::Axial:
    width = mif.getColumnCount();
    height = mif.getRowCount();
    pixelWidth = mif.getChannel(1).getPixelSizeX();
    pixelHeight = mif.getChannel(1).getPixelSizeY();
    pixelDepth = mif.getChannel(1).getSliceThickness();
    if (getUseNormalizedIndex())
      slicePos = getSlicePosition() * mif.getSliceCount() + 0.5;
    else
      slicePos = getSlicePosition();
    slicePos = max(0, min(slicePos, mif.getSliceCount() - 1));
    break;
  case SliceOrientation::Sagital:
    width = mif.getRowCount();
    height = mif.getSliceCount();
    pixelWidth = mif.getChannel(1).getPixelSizeY();
    pixelHeight = mif.getChannel(1).getSliceThickness();
    pixelDepth = mif.getChannel(1).getPixelSizeX();
    if (getUseNormalizedIndex())
      slicePos = getSlicePosition() * mif.getColumnCount() + 0.5;
    else
      slicePos = getSlicePosition();
    slicePos = max(0, min(slicePos, mif.getColumnCount() - 1));
    break;
  case SliceOrientation::Coronal:
    width = mif.getColumnCount();
    height = mif.getSliceCount();
    pixelWidth = mif.getChannel(1).getPixelSizeX();
    pixelHeight = mif.getChannel(1).getSliceThickness();
    pixelDepth = mif.getChannel(1).getPixelSizeY();
    if (getUseNormalizedIndex())
      slicePos = getSlicePosition() * mif.getRowCount() + 0.5;
    else
      slicePos = getSlicePosition();
    slicePos = max(0, min(slicePos, mif.getRowCount() - 1));
    break;
  }

  boost::shared_ptr<image_t> image(new image_t(width, height, 1,
      pixelWidth, pixelHeight, pixelDepth));
  float* buffer = image->getData();

  CMIF::pixelArray pixels = mif.getRawData();
  for (int y = 0, i = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x, ++i) {
      switch (getOrientation()) {
      case SliceOrientation::Axial:
        buffer[i] = (float)pixels[slicePos + 1][y][x] / (float)getMaximumIntensity();
        break;
      case SliceOrientation::Sagital:
        buffer[i] = (float)pixels[mif.getSliceCount() - y][x][slicePos] / (float)getMaximumIntensity();
        break;
      case SliceOrientation::Coronal:
        buffer[i] = (float)pixels[mif.getSliceCount() - y][slicePos][x] / (float)getMaximumIntensity();
        break;
      }

    }
  }

  newState->setWidth(width);
  newState->setHeight(height);
  newState->setImage(image);
}

}

}

}
