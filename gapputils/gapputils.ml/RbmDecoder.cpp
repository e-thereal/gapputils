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

BeginPropertyDefinitions(RbmDecoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(RbmModel, Input("RBM"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenVector, Input("In"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleVector, Output("Out"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(IsGaussian, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(UseWeightsOnly, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

RbmDecoder::RbmDecoder() : _IsGaussian(0), _UseWeightsOnly(0), data(0) {
  WfeUpdateTimestamp
  setLabel("RbmDecoder");

  Changed.connect(capputils::EventHandler<RbmDecoder>(this, &RbmDecoder::changedHandler));
}

RbmDecoder::~RbmDecoder() {
  if (data)
    delete data;
}

void RbmDecoder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmDecoder::writeResults() {
  if (!data)
    return;

  setVisibleVector(data->getVisibleVector());
}

}

}
