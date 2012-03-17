/*
 * Mean.cpp
 *
 *  Created on: Mar 13, 2012
 *      Author: tombr
 */

#include "Mean.h"

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

BeginPropertyDefinitions(Mean)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputVectors, Input(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))
  DefineProperty(FeatureCount, Observe(PROPERTY_ID))
  DefineProperty(OutputVector, Output(""), Volatile(), ReadOnly(), Observe(PROPERTY_ID))

EndPropertyDefinitions

Mean::Mean() : _FeatureCount(0), data(0) {
  WfeUpdateTimestamp
  setLabel("Mean");

  Changed.connect(capputils::EventHandler<Mean>(this, &Mean::changedHandler));
}

Mean::~Mean() {
  if (data)
    delete data;
}

void Mean::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Mean::writeResults() {
  if (!data)
    return;

  setOutputVector(data->getOutputVector());
}

}

}
