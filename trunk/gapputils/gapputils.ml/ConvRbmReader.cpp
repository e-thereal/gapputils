/*
 * ConvRbmReader.cpp
 *
 *  Created on: Apr 09, 2012
 *      Author: tombr
 */

#include "ConvRbmReader.h"

#include <capputils/EventHandler.h>
#include <capputils/FileExists.h>
#include <capputils/FilenameAttribute.h>
#include <capputils/InputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/Serializer.h>
#include <capputils/TimeStampAttribute.h>
#include <capputils/Verifier.h>
#include <capputils/VolatileAttribute.h>
#include <capputils/NoParameterAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(ConvRbmReader)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Input("File"), Filename("CRBM Model (*.crbm)"), FileExists(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Model, Output("CRBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(FilterCount, NoParameter(), Observe(PROPERTY_ID))
  DefineProperty(FilterSize, NoParameter(), Observe(PROPERTY_ID))
  DefineProperty(PoolingSize, NoParameter(), Observe(PROPERTY_ID))

EndPropertyDefinitions

ConvRbmReader::ConvRbmReader() : data(0) {
  WfeUpdateTimestamp
  setLabel("Reader");

  Changed.connect(capputils::EventHandler<ConvRbmReader>(this, &ConvRbmReader::changedHandler));
}

ConvRbmReader::~ConvRbmReader() {
  if (data)
    delete data;
}

void ConvRbmReader::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void ConvRbmReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new ConvRbmReader();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<ConvRbmModel> crbm(new ConvRbmModel());
  capputils::Serializer::readFromFile(*crbm, getFilename());

  data->setModel(crbm);
  data->setFilterCount(crbm->getFilters()->size());
  data->setFilterSize(crbm->getFilters()->at(1)->size()[0]);
  data->setPoolingSize(crbm->getPoolingBlockSize());
}

void ConvRbmReader::writeResults() {
  if (!data)
    return;

  setModel(data->getModel());
  setFilterCount(data->getFilterCount());
  setFilterSize(data->getFilterSize());
  setPoolingSize(data->getPoolingSize());
}

}

}
