/*
 * FgrbmEncoder.cpp
 *
 *  Created on: Dec 25, 2011
 *      Author: tombr
 */

#include "FgrbmEncoder.h"

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

BeginPropertyDefinitions(FgrbmEncoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(FgrbmModel, Input("FGRBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ConditionalVector, Input("Cond"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleVector, Input("In"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenVector, Output("Out"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(SampleHiddens, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(IsGaussian, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FgrbmEncoder::FgrbmEncoder() : _SampleHiddens(true), _IsGaussian(false), data(0) {
  WfeUpdateTimestamp
  setLabel("FgrbmEncoder");

  Changed.connect(capputils::EventHandler<FgrbmEncoder>(this, &FgrbmEncoder::changedHandler));
}

FgrbmEncoder::~FgrbmEncoder() {
  if (data)
    delete data;
}

void FgrbmEncoder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void FgrbmEncoder::writeResults() {
  if (!data)
    return;

  setHiddenVector(data->getHiddenVector());
}

}

}
