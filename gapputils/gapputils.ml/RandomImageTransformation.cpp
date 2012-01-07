/*
 * RandomImageTransformation.cpp
 *
 *  Created on: Nov 30, 2011
 *      Author: tombr
 */

#include "RandomImageTransformation.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RandomImageTransformation)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Input, Input("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(RowCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(XRange, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(YRange, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Output, Output("Img"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RandomImageTransformation::RandomImageTransformation() : _RowCount(1), _ColumnCount(1), data(0) {
  WfeUpdateTimestamp
  setLabel("RandomImageTransformation");

  _XRange.resize(2);
  _XRange[0] = -1;
  _XRange[1] = 1;
  _YRange.resize(2);
  _YRange[0] = -1;
  _YRange[1] = 1;

  Changed.connect(capputils::EventHandler<RandomImageTransformation>(this, &RandomImageTransformation::changedHandler));
}

RandomImageTransformation::~RandomImageTransformation() {
  if (data)
    delete data;
}

void RandomImageTransformation::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RandomImageTransformation::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RandomImageTransformation();

  if (!capputils::Verifier::Valid(*this))
    return;

  if (!getInput() || getRowCount() < 1 || getColumnCount() < 1 || getXRange().size() != 2 || getYRange().size() != 2)
    return;

  std::vector<double>& input = *getInput();

  const int rowCount = getRowCount();
  const int columnCount = getColumnCount();
  const int featureCount = rowCount * columnCount;
  const int count = (int)input.size();
  const std::vector<int>& xrange = getXRange();
  const std::vector<int>& yrange = getYRange();

  if (count % featureCount != 0)
    return;

  boost::shared_ptr<std::vector<double> > output(new std::vector<double>(count));
  for (int sliceOffset = 0; sliceOffset < count; sliceOffset += featureCount) {
    // Get random transformation parameters
    // apply transformation

    int dx = rand() % (xrange[1] - xrange[0] + 1) + xrange[0];
    int dy = rand() % (yrange[1] - yrange[0] + 1) + yrange[0];

    std::copy(&input[std::max(0, sliceOffset - dx - dy * columnCount)],
        &input[std::min(count - 1, sliceOffset - dx - dy * columnCount + featureCount)],
        output->begin() + sliceOffset);
  }

  data->setOutput(output);
}

void RandomImageTransformation::writeResults() {
  if (!data)
    return;

  setOutput(data->getOutput());
}

}

}
