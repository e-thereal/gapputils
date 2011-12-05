/*
 * BinaryImageGenerator.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "BinaryImageGenerator.h"

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
#include <cstdlib>
#include <iostream>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(BinaryImageGenerator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(RowCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ColumnCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ImageCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(FeatureCount, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Data, Output("Imgs"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

BinaryImageGenerator::BinaryImageGenerator()
 : _RowCount(0), _ColumnCount(0), _ImageCount(0), _FeatureCount(0), data(0)
{
  WfeUpdateTimestamp
  setLabel("BinaryImageGenerator");

  Changed.connect(capputils::EventHandler<BinaryImageGenerator>(this, &BinaryImageGenerator::changedHandler));
}

BinaryImageGenerator::~BinaryImageGenerator() {
  if (data)
    delete data;
}

void BinaryImageGenerator::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void BinaryImageGenerator::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new BinaryImageGenerator();

  if (!capputils::Verifier::Valid(*this))
    return;

  const int featureCount = getRowCount() * getColumnCount();
  const int count = getImageCount() * featureCount;
  if (count < 0)
    return;

  using namespace boost::lambda;

  boost::shared_ptr<std::vector<float> > randomData(new std::vector<float>(count));
  for (int i = 0; i < count; ++i)
     randomData->at(i) = ((rand() % 10) == 0 ? 1.f : 0.f);

  data->setFeatureCount(featureCount);
  data->setData(randomData);
}

void BinaryImageGenerator::writeResults() {
  if (!data)
    return;

  setFeatureCount(data->getFeatureCount());
  setData(data->getData());
}

}

}
