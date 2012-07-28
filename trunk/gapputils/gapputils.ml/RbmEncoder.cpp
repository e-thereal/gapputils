/*
 * RbmEncoder.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmEncoder.h"

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

namespace ml {

BeginPropertyDefinitions(RbmEncoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(RbmModel, Input("RBM"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(VisibleVector, Input("In"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(HiddenVector, Output("Out"), ReadOnly(), Volatile(), Observe(Id), TimeStamp(Id))
  DefineProperty(SampleHiddens, Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

RbmEncoder::RbmEncoder() : _SampleHiddens(true), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmEncoder");

  Changed.connect(capputils::EventHandler<RbmEncoder>(this, &RbmEncoder::changedHandler));
}

RbmEncoder::~RbmEncoder() {
  if (data)
    delete data;
}

void RbmEncoder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmEncoder::writeResults() {
  if (!data)
    return;

  setHiddenVector(data->getHiddenVector());
}

}

}
