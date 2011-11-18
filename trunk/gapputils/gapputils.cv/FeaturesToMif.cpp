/*
 * PrincipleComponentsToMif.cpp
 *
 *  Created on: Jun 10, 2011
 *      Author: tombr
 */

#include "FeaturesToMif.h"

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

#include <algorithm>
#include <cmath>

#include <CMIF.hpp>
#include <CSlice.hpp>
#include <CProcessInfo.hpp>
#include <CChannel.hpp>
#include <iter/CPixelIterators.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(FeaturesToMif)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Data, Input(), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RowCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MaxCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MifName, Output("Mif"), Filename(), NotEqual<std::string>(""), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FeaturesToMif::FeaturesToMif() : _ColumnCount(1), _RowCount(1), _MaxCount(-1), data(0) {
  WfeUpdateTimestamp
  setLabel("FeaturesToMif");

  static char** argv = new char*[1];
  argv[0] = "FeaturesToMif";
  MSMRI::CProcessInfo::getInstance().getCommandLine(1, argv);

  Changed.connect(capputils::EventHandler<FeaturesToMif>(this, &FeaturesToMif::changedHandler));
}

FeaturesToMif::~FeaturesToMif() {
  if (data)
    delete data;
}

void FeaturesToMif::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FeaturesToMif::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;

  if (!data)
    data = new FeaturesToMif();

  if (!capputils::Verifier::Valid(*this) || !getData())
    return;

  //const int count = getColumnCount() * getRowCount() * getSliceCount();

  const int columnCount = getColumnCount();
  const int rowCount = getRowCount();
  const int totalSliceCount = getData()->size() / (columnCount * rowCount);
  const int sliceCount = (getMaxCount() == -1 ? totalSliceCount : std::min(getMaxCount(), totalSliceCount));

  CMIF mif(columnCount, rowCount, sliceCount);
  CMIF::pixelArray pixels = mif.getRawData();

  std::vector<float>& features = *getData().get();

  float minV = features[0], maxV = features[0];

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

  for (int z = 1, i = 0; z <= mif.getSliceCount(); ++z) {
    for (int y = 0; y < mif.getRowCount(); ++y) {
      for (int x = 0; x < mif.getColumnCount(); ++x, ++i) {
        pixels[z][y][x] = (features[i] - minV) * 255. / (maxV - minV);
      }
    }
    if (monitor)
      monitor->reportProgress(50 * z / mif.getSliceCount() + 50);
  }

  mif.writeToFile(getMifName(), true);
}

void FeaturesToMif::writeResults() {
  if (!data)
    return;

  setMifName(getMifName());
}

}

}

