/*
 * RbmReader.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: tombr
 */

#include "RbmReader.h"

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

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(RbmReader)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Filename, Input("File"), Filename("RBM Model (*.rbm)"), FileExists(), Observe(Id), TimeStamp(Id))
  DefineProperty(RbmModel, Output("RBM"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(VisibleCount, Observe(Id), TimeStamp(Id))
  DefineProperty(HiddenCount, Observe(Id), TimeStamp(Id))
  DefineProperty(HiddenUnitType, Observe(Id))

EndPropertyDefinitions

RbmReader::RbmReader() : _VisibleCount(0), _HiddenCount(0), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmReader");

  Changed.connect(capputils::EventHandler<RbmReader>(this, &RbmReader::changedHandler));
}

RbmReader::~RbmReader() {
  if (data)
    delete data;
}

void RbmReader::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmReader::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new RbmReader();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<RbmModel> rbm(new RbmModel());
  capputils::Serializer::readFromFile(*rbm, getFilename());

  data->setRbmModel(rbm);
  data->setVisibleCount(rbm->getVisibleBiases()->size());
  data->setHiddenCount(rbm->getHiddenBiases()->size());
  data->setHiddenUnitType(rbm->getHiddenUnitType());
}

void RbmReader::writeResults() {
  if (!data)
    return;

  setRbmModel(data->getRbmModel());
  setVisibleCount(data->getVisibleCount());
  setHiddenCount(data->getHiddenCount());
  setHiddenUnitType(data->getHiddenUnitType());
}

}

}
