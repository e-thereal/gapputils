/*
 * MifReader.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: tombr
 */

#include "MifReader.h"

#include <CMIF.hpp>
#include <CChannel.hpp>
#include <cmath>

#include <capputils/attributes/RenamedAttribute.h>
#include <capputils/attributes/DeprecatedAttribute.h>

namespace gml {

namespace imaging {

namespace io {

BeginPropertyDefinitions(MifReader, Renamed("gml::imaging::io::OpenMif"), Deprecated("Use OpenMif instead."))

  ReflectableBase(DefaultWorkflowElement<MifReader>)

  WorkflowProperty(MifName, Input("Mif"), Filename("MIFs (*.MIF)"), FileExists())
  WorkflowProperty(Image, Output("Img"))
  WorkflowProperty(MaximumIntensity)
  WorkflowProperty(Width, NoParameter())
  WorkflowProperty(Height, NoParameter())
  WorkflowProperty(Depth, NoParameter())
  WorkflowProperty(VoxelWidth, NoParameter(), Description("Voxel width in mm."))
  WorkflowProperty(VoxelHeight, NoParameter(), Description("Voxel height in mm."))
  WorkflowProperty(VoxelDepth, NoParameter(), Description("Voxel depth in mm."))

EndPropertyDefinitions

MifReader::MifReader() : _MaximumIntensity(2048), _Width(0), _Height(0), _Depth(0), _VoxelWidth(1), _VoxelHeight(1), _VoxelDepth(1)
{
  setLabel("Mif");
}

void MifReader::update(IProgressMonitor* /*monitor*/) const {
  using namespace MSMRI::MIF;
  using namespace std;

  Logbook& dlog = getLogbook();
  dlog.setSeverity(Severity::Trace);

  CMIF mif(getMifName());

  int width, height, depth;

  size_t pixelWidth = 1000, pixelHeight = 1000, pixelDepth = 1000;

  width = mif.getColumnCount();
  height = mif.getRowCount();
  depth = mif.getSliceCount();
  pixelWidth = mif.getChannel(1).getPixelSizeX();
  pixelHeight = mif.getChannel(1).getPixelSizeY();
  pixelDepth = mif.getChannel(1).getSliceThickness();

  boost::shared_ptr<image_t> image(new image_t(width, height, depth,
      pixelWidth, pixelHeight, pixelDepth));
  float* buffer = image->getData();

  float maxValue = getMaximumIntensity();

  CMIF::pixelArray pixels = mif.getRawData();
  for (int z = 1, i = 0; z <= depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x, ++i) {
        buffer[i] = (float)pixels[z][y][x] / maxValue;
      }
    }
  }

  newState->setWidth(width);
  newState->setHeight(height);
  newState->setDepth(depth);
  newState->setVoxelWidth((double)pixelWidth / 1000.0);
  newState->setVoxelHeight((double)pixelHeight / 1000.0);
  newState->setVoxelDepth((double)pixelDepth / 1000.0);
  newState->setImage(image);
}

}

}

}
