/*
 * SliceToFeatures.cpp
 *
 *  Created on: Jun 8, 2011
 *      Author: tombr
 */

#include "SliceToFeatures.h"

#include <algorithm>
#include <iostream>

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
#include <capputils/EnumerableAttribute.h>
#include <capputils/NoParameterAttribute.h>

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

BeginPropertyDefinitions(SliceToFeatures)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(MifNames, Input("Mifs"), Filename("MIFs (*.MIF);;ROI MIFs (*_roi.MIF)", true), Enumerable<std::vector<std::string>, false>(), FileExists(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RowCount, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VoxelsPerSlice, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SliceCount, NoParameter(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Data, Output(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID), Volatile(), ReadOnly())

EndPropertyDefinitions

SliceToFeatures::SliceToFeatures() : _RowCount(0), _ColumnCount(0), _VoxelsPerSlice(0), _SliceCount(0), data(0) {
  WfeUpdateTimestamp
  setLabel("SliceToFeatures");

  Changed.connect(capputils::EventHandler<SliceToFeatures>(this, &SliceToFeatures::changedHandler));
}

SliceToFeatures::~SliceToFeatures() {
  if (data)
    delete data;
}

void SliceToFeatures::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void SliceToFeatures::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  using namespace MSMRI::MIF;
  using namespace std;

  cout << "Checking validity of model." << endl;

  if (!data)
    data = new SliceToFeatures();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (_MifNames.size() == 0)
    return;

  cout << "Start getting features." << endl;

  CMIF firstMif(_MifNames[0]);

  const int columnCount = firstMif.getColumnCount();
  const int rowCount = firstMif.getRowCount();
  const int voxelsPerSlice = rowCount * columnCount;

  boost::shared_ptr<std::vector<float> > sliceData(new std::vector<float>());

  for (unsigned i = 0; i < _MifNames.size(); ++i) {

    CMIF mif(_MifNames[i]);

    if (columnCount != mif.getColumnCount() || rowCount != mif.getRowCount())
      continue;

    const int sliceCount = mif.getSliceCount();
    const int oldSize = sliceData->size();
    sliceData->resize(oldSize + voxelsPerSlice * sliceCount);
    std::copy(mif.beginPixels(), mif.endPixels(), sliceData->begin() + oldSize);
  }

  data->setRowCount(rowCount);
  data->setColumnCount(columnCount);
  if (voxelsPerSlice)
    data->setSliceCount(sliceData->size() / voxelsPerSlice);
  data->setData(sliceData);
  data->setVoxelsPerSlice(voxelsPerSlice);

  cout << "Done getting features." << endl;
}

void SliceToFeatures::writeResults() {
  if (!data)
    return;

  setRowCount(data->getRowCount());
  setColumnCount(data->getColumnCount());
  setVoxelsPerSlice(data->getVoxelsPerSlice());
  setSliceCount(data->getSliceCount());
  setData(data->getData());
}

}

}
