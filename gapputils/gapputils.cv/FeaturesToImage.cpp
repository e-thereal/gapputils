/*
 * FeaturesToImage.cpp
 *
 *  Created on: Dec 08, 2011
 *      Author: tombr
 */

#include "FeaturesToImage.h"

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

#include <culib/CudaImage.h>

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

int FeaturesToImage::dataId;

BeginPropertyDefinitions(FeaturesToImage)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Data, Input(), ReadOnly(), Volatile(), Observe(dataId = PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RowCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(MaxCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Image, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FeaturesToImage::FeaturesToImage()
 : _ColumnCount(1), _RowCount(1), _MaxCount(-1), data(0)
{
  WfeUpdateTimestamp
  setLabel("F2I");

  Changed.connect(capputils::EventHandler<FeaturesToImage>(this, &FeaturesToImage::changedHandler));
}

FeaturesToImage::~FeaturesToImage() {
  if (data)
    delete data;
}

void FeaturesToImage::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == dataId) {
    execute(0);
    writeResults();
  }
}

void FeaturesToImage::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FeaturesToImage();

  if (!capputils::Verifier::Valid(*this) || !getData())
    return;

  const int columnCount = getColumnCount();
  const int rowCount = getRowCount();
  const int totalSliceCount = getData()->size() / (columnCount * rowCount);
  const int sliceCount = (getMaxCount() == -1 ? totalSliceCount : std::min(getMaxCount(), totalSliceCount));

  std::vector<float>& features = *getData();

  boost::shared_ptr<culib::ICudaImage> image(new culib::CudaImage(dim3(columnCount, rowCount, sliceCount)));
  std::copy(features.begin(), features.begin() + (rowCount * columnCount * sliceCount), image->getOriginalImage());
  image->resetWorkingCopy();
  data->setImage(image);
}

void FeaturesToImage::writeResults() {
  if (!data)
    return;

  setImage(data->getImage());
}

}

}

