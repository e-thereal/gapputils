/*
 * ConvRbmDecoder.cpp
 *
 *  Created on: Apr 9, 2012
 *      Author: tombr
 */

#include "ConvRbmDecoder.h"

#include <capputils/DeprecatedAttribute.h>
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

int ConvRbmDecoder::inputId;

BeginPropertyDefinitions(ConvRbmDecoder, Deprecated("Use gapputils::ml::ConvRbmEncoder instead."))

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Model, Input("CRBM"), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(Inputs, Input("H"), Volatile(), ReadOnly(), Observe(inputId = Id))
  DefineProperty(Outputs, Output("V"), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(SampleVisibles, Observe(Id))
  DefineProperty(Auto, Observe(Id))

EndPropertyDefinitions

ConvRbmDecoder::ConvRbmDecoder() : _SampleVisibles(false), _Auto(false), data(0) {
  WfeUpdateTimestamp
  setLabel("ConvRbmDecoder");

  Changed.connect(capputils::EventHandler<ConvRbmDecoder>(this, &ConvRbmDecoder::changedHandler));
}

ConvRbmDecoder::~ConvRbmDecoder() {
  if (data)
    delete data;
}

void ConvRbmDecoder::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == inputId && getAuto()) {
    execute(0);
    writeResults();
  }
}

void ConvRbmDecoder::writeResults() {
  if (!data)
    return;

  setOutputs(data->getOutputs());
}

}

}
