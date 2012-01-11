/*
 * DV2FV.cpp
 *
 *  Created on: Jan 10, 2012
 *      Author: tombr
 */

#include "DV2FV.h"

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

namespace common {

BeginPropertyDefinitions(DV2FV)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Input, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Output, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

DV2FV::DV2FV() : data(0) {
  WfeUpdateTimestamp
  setLabel("DV2FV");

  Changed.connect(capputils::EventHandler<DV2FV>(this, &DV2FV::changedHandler));
}

DV2FV::~DV2FV() {
  if (data)
    delete data;
}

void DV2FV::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void DV2FV::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new DV2FV();

  if (!capputils::Verifier::Valid(*this) || !getInput())
    return;

  boost::shared_ptr<std::vector<float> > output(new std::vector<float>(getInput()->size()));
  std::copy(getInput()->begin(), getInput()->end(), output->begin());
  data->setOutput(output);
}

void DV2FV::writeResults() {
  if (!data)
    return;

  setOutput(data->getOutput());
}

}

}
