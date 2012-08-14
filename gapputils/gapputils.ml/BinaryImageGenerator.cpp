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
#include <capputils/NoParameterAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>

#include <culib/CudaImage.h>

#include "distributions.h"

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(BinaryImageGenerator)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(RowCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ColumnCount, Observe(Id), TimeStamp(Id))
  DefineProperty(ImageCount, Observe(Id), TimeStamp(Id))
  DefineProperty(IsBinary, Observe(Id), TimeStamp(Id))
  DefineProperty(Density, Observe(Id))
  DefineProperty(FeatureCount, NoParameter(), Observe(Id), TimeStamp(Id))
  DefineProperty(Data, Output("Imgs"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

BinaryImageGenerator::BinaryImageGenerator()
 : _RowCount(0), _ColumnCount(0), _ImageCount(0), _IsBinary(true), _Density(10), _FeatureCount(0), data(0)
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
  const int density = getDensity();
  if (count < 0)
    return;

  using namespace boost::lambda;

  boost::shared_ptr<std::vector<double> > randomData(new std::vector<double>(count));
  createNormalSample normals;
  if (getIsBinary()) {
    for (int i = 0; i < count; ++i) {
       randomData->at(i) = ((rand() % density) == 0 ? 1.f : 0.f);
    }
  } else {
    for (int i = 0; i < count; ++i) {
       randomData->at(i) = normals(0.0);
    }
  }

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
