/*
 * RbmConditional.cpp
 *
 *  Created on: Mar 12, 2012
 *      Author: tombr
 */

#include "RbmConditional.h"

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

BeginPropertyDefinitions(RbmConditional)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Model, Input("RBM"), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(Givens, Input("X"), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(Conditionals, Output("Y"), Volatile(), ReadOnly(), Observe(Id))
  DefineProperty(GivenCount, Observe(Id))
  DefineProperty(InitializationCycles, Observe(Id))
  DefineProperty(SampleCycles, Observe(Id))
  DefineProperty(ShowSamples, Observe(Id))
  DefineProperty(Delay, Observe(Id))
  DefineProperty(Debug, Observe(Id))

EndPropertyDefinitions

RbmConditional::RbmConditional()
 : _GivenCount(1), _InitializationCycles(10), _SampleCycles(10), _ShowSamples(false),
   _Delay(0), _Debug(0), data(0)
{
  WfeUpdateTimestamp
  setLabel("RbmConditional");

  Changed.connect(capputils::EventHandler<RbmConditional>(this, &RbmConditional::changedHandler));
}

RbmConditional::~RbmConditional() {
  if (data)
    delete data;
}

void RbmConditional::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void RbmConditional::writeResults() {
  if (!data)
    return;

  setConditionals(data->getConditionals());
}

}

}
