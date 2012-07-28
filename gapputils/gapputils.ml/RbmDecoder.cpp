/*
 * RbmDecoder.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "RbmDecoder.h"

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

int RbmDecoder::inputId;

BeginPropertyDefinitions(RbmDecoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(RbmModel, Input("RBM"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(HiddenVector, Input("In"), Volatile(), ReadOnly(), Observe(inputId = Id), TimeStamp(Id))
  DefineProperty(VisibleVector, Output("Out"), Volatile(), ReadOnly(), Observe(Id), TimeStamp(Id))
  DefineProperty(UseWeightsOnly, Observe(Id), TimeStamp(Id))
  DefineProperty(AutoDecode, Observe(Id))

EndPropertyDefinitions

RbmDecoder::RbmDecoder() : _UseWeightsOnly(0), _AutoDecode(false), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmDecoder");

  Changed.connect(capputils::EventHandler<RbmDecoder>(this, &RbmDecoder::changedHandler));
}

RbmDecoder::~RbmDecoder() {
  if (data)
    delete data;
}

void RbmDecoder::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAutoDecode()) {
    execute(0);
    writeResults();
  }
}

void RbmDecoder::writeResults() {
  if (!data)
    return;

  setVisibleVector(data->getVisibleVector());
}

}

}
