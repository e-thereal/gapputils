/*
 * FV2DV.cpp
 *
 *  Created on: Jan 16, 2012
 *      Author: tombr
 */

#include "FV2DV.h"

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

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

int FV2DV::inputId;

BeginPropertyDefinitions(FV2DV)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Input, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(inputId = PROPERTY_ID))
  DefineProperty(Output, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FV2DV::FV2DV() : _Auto(false), data(0) {
  WfeUpdateTimestamp
  setLabel("FV2DV");

  Changed.connect(capputils::EventHandler<FV2DV>(this, &FV2DV::changedHandler));
}

FV2DV::~FV2DV() {
  if (data)
    delete data;
}

void FV2DV::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAuto()) {
    execute(0);
    writeResults();
  }
}

void FV2DV::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new FV2DV();

  if (!capputils::Verifier::Valid(*this) || !getInput())
    return;

  boost::shared_ptr<std::vector<double> > output(new std::vector<double>(getInput()->size()));
  std::copy(getInput()->begin(), getInput()->end(), output->begin());
  data->setOutput(output);
}

void FV2DV::writeResults() {
  if (!data)
    return;

  setOutput(data->getOutput());
}

}

}
