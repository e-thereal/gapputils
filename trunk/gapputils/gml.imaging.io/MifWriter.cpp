/*
 * MifWriter.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#include "MifWriter.h"

#include <CMIF.hpp>
#include <CSlice.hpp>
#include <CProcessInfo.hpp>
#include <CChannel.hpp>
#include <iter/CPixelIterators.hpp>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(MifWriter)
  ReflectableBase(DefaultWorkflowElement<MifWriter>)

  WorkflowProperty(Image, Input("Img"))
  WorkflowProperty(MifName, Filename(), NotEmpty<Type>())
  WorkflowProperty(MinValue)
  WorkflowProperty(MaxValue)
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(AutoScale)
  WorkflowProperty(OutputName, Output("Mif"))

EndPropertyDefinitions

MifWriter::MifWriter() : _MinValue(0), _MaxValue(1), _MaximumIntensity(2048), _AutoScale(true) {
  setLabel("MifWriter");

  static char** argv = new char*[1];
  static char progName[] = "MifWriter";
  argv[0] = progName;
  MSMRI::CProcessInfo::getInstance().getCommandLine(1, argv);
}

void MifWriter::update(IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;

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
  newState->setOutputName(getMifName());
}

}

}

}
