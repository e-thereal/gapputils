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

#include <algorithm>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(MifWriter)
  ReflectableBase(DefaultWorkflowElement<MifWriter>)

  WorkflowProperty(Image, Input("I"))
  WorkflowProperty(Images, Input("Is"))
  WorkflowProperty(MifName, Filename("MIFs (*.MIF)"), NotEmpty<Type>())
  WorkflowProperty(MinValue)
  WorkflowProperty(MaxValue)
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(AutoScale, Description("One intensity window is calculated for all images."))
  WorkflowProperty(OutputName, Output("Mif"))

EndPropertyDefinitions

MifWriter::MifWriter() : _MinValue(0), _MaxValue(1), _MaximumIntensity(2048), _AutoScale(false) {
  setLabel("MifWriter");

  static char** argv = new char*[1];
  static char progName[] = "MifWriter";
  argv[0] = progName;
  MSMRI::CProcessInfo::getInstance().getCommandLine(1, argv);
}

void MifWriter::update(IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;
  Logbook& dlog = getLogbook();

  std::vector<boost::shared_ptr<image_t> > images;

  if (getImage())
    images.push_back(getImage());

  if (getImages()) {
    const size_t oldSize = images.size();
    images.resize(oldSize + getImages()->size());
    std::copy(getImages()->begin(), getImages()->end(), images.begin() + oldSize);
  }

  if (!images.size()) {
    dlog(Severity::Warning) << "No images given. Aborting!";
    return;
  }

  const int columnCount = images[0]->getSize()[0];
  const int rowCount = images[0]->getSize()[1];
  const int sliceCount = images[0]->getSize()[2];

  double maxIntens = getMaximumIntensity();
  float minV = getMinValue(), maxV = getMaxValue();

  CMIF mif(columnCount, rowCount, sliceCount, images.size());

  if (getAutoScale()) {
    for (size_t iImage = 1; iImage <= images.size(); ++iImage) {
      float* features = images[iImage - 1]->getData();

      if (iImage == 1)
        minV = features[0], maxV = features[0];
      for (int z = 1, i = 0; z <= mif.getSliceCount(); ++z) {
        for (int y = 0; y < mif.getRowCount(); ++y) {
          for (int x = 0; x < mif.getColumnCount(); ++x, ++i) {
            minV = std::min(minV, features[i]);
            maxV = std::max(maxV, features[i]);
          }
        }
        if (monitor)
          monitor->reportProgress(45.0 * ((double)z / mif.getSliceCount() + iImage - 1) / images.size());
      }
    }
  }

  for (size_t iImage = 1; iImage <= images.size(); ++iImage) {

    image_t& image = *images[iImage - 1];

    mif.getChannel(iImage).setPixelSizeX(image.getPixelSize()[0]);
    mif.getChannel(iImage).setPixelSizeY(image.getPixelSize()[1]);
    mif.getChannel(iImage).setSliceThickness(image.getPixelSize()[2]);
    CMIF::pixelArray pixels = mif.getRawData(iImage);

    float* features = image.getData();

    for (int z = 1, i = 0; z <= mif.getSliceCount(); ++z) {
      for (int y = 0; y < mif.getRowCount(); ++y) {
        for (int x = 0; x < mif.getColumnCount(); ++x, ++i) {
          pixels[z][y][x] = std::min(4096.0, std::max(0.0, (features[i] - minV) * maxIntens / (maxV - minV)));
        }
      }
      if (monitor) {
        if (getAutoScale()) {
          monitor->reportProgress(45.0 * ((double)z / mif.getSliceCount() + iImage - 1) / images.size() + 45.0);
        } else {
          monitor->reportProgress(90.0 * ((double)z / mif.getSliceCount() + iImage - 1) / images.size());
        }
      }
    }
  }

  mif.writeToFile(getMifName(), true);
  newState->setOutputName(getMifName());
}

}

}

}
