/*
 * FgrbmDecoder.cpp
 *
 *  Created on: Jan 10, 2012
 *      Author: tombr
 */

#include "FgrbmDecoder.h"

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

BeginPropertyDefinitions(FgrbmDecoder)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(FgrbmModel, Input("FGRBM"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(ConditionalVector, Input("Cond"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(HiddenVector, Input("In"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(VisibleVector, Output("Out"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  //DefineProperty(SampleHiddens, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  //DefineProperty(IsGaussian, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

FgrbmDecoder::FgrbmDecoder() : data(0) {
  WfeUpdateTimestamp
  setLabel("FgrbmDecoder");

  Changed.connect(capputils::EventHandler<FgrbmDecoder>(this, &FgrbmDecoder::changedHandler));
}

FgrbmDecoder::~FgrbmDecoder() {
  if (data)
    delete data;
}

void FgrbmDecoder::changedHandler(capputils::ObservableClass* sender, int eventId) {

}


void FgrbmDecoder::writeResults() {
  if (!data)
    return;

  setVisibleVector(data->getVisibleVector());
}

}

}
