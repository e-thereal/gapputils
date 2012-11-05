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
  DefineProperty(Filename, Input("File"), Filename("CRBM Model (*.crbm)"), FileExists(), Observe(Id), TimeStamp(Id))
  DefineProperty(Model, Output("CRBM"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(FilterCount, NoParameter(), Observe(Id))
  DefineProperty(FilterWidth, NoParameter(), Observe(Id))
  DefineProperty(FilterHeight, NoParameter(), Observe(Id))
  DefineProperty(FilterDepth, NoParameter(), Observe(Id))
  DefineProperty(PoolingSize, NoParameter(), Observe(Id))
  DefineProperty(HiddenUnitType, NoParameter(), Observe(Id))

EndPropertyDefinitions

ConvRbmReader::ConvRbmReader() : _FilterWidth(0), _FilterHeight(0), _FilterDepth(0), data(0) {
  WfeUpdateTimestamp
  setLabel("Reader");
}

ConvRbmReader::~ConvRbmReader() {
  if (data)
    delete data;
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
  if (crbm->getFilters()->size()) {
    data->setFilterWidth(crbm->getFilters()->at(0)->size()[0]);
    data->setFilterHeight(crbm->getFilters()->at(0)->size()[1]);
    data->setFilterDepth(crbm->getFilters()->at(0)->size()[2]);
  }
  data->setPoolingSize(crbm->getPoolingBlockSize());
  data->setHiddenUnitType(crbm->getHiddenUnitType());
}

void ConvRbmReader::writeResults() {
  if (!data)
    return;

  setModel(data->getModel());
  setFilterCount(data->getFilterCount());
  setFilterWidth(data->getFilterWidth());
  setFilterHeight(data->getFilterHeight());
  setFilterDepth(data->getFilterDepth());
  setPoolingSize(data->getPoolingSize());
  setHiddenUnitType(data->getHiddenUnitType());
}

}

}
