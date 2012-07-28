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
#include <gapputils/ReadOnlyAttribute.h>

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
  DefineProperty(Data, Input(), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(MaxCount, Observe(Id), TimeStamp(Id))
  DefineProperty(MinValue, Observe(Id), TimeStamp(Id))
  DefineProperty(MaxValue, Observe(Id), TimeStamp(Id))
  DefineProperty(AutoScale, Observe(Id), TimeStamp(Id))
  DefineProperty(MifName, Output("Mif"), Filename(), NotEqual<std::string>(""), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

FeaturesToMif::FeaturesToMif()
 : _ColumnCount(1), _RowCount(1), _MaxCount(-1), _MinValue(0), _MaxValue(1),
   _AutoScale(true), data(0)
{
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

  const int columnCount = getColumnCount();
  const int rowCount = getRowCount();
  const int totalSliceCount = getData()->size() / (columnCount * rowCount);
  const int sliceCount = (getMaxCount() == -1 ? totalSliceCount : std::min(getMaxCount(), totalSliceCount));

  CMIF mif(columnCount, rowCount, sliceCount);
  CMIF::pixelArray pixels = mif.getRawData();

  std::vector<float>& features = *getData().get();

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
        pixels[z][y][x] = std::min(255.0, std::max(0.0, (features[i] - minV) * 255. / (maxV - minV)));
//        pixels[z][y][x] = (features[i] - minV) * 255. / (maxV - minV);
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

void FeaturesToMif::writeResults() {
  if (!data)
    return;

  setMifName(getMifName());
}

}

}

