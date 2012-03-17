/*
 * TensorTest.cpp
 *
 *  Created on: Mar 7, 2012
 *      Author: tombr
 */

#include "TensorTest.h"

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

BeginPropertyDefinitions(TensorTest)

  ReflectableBase(gapputils::workflow::WorkflowElement)

EndPropertyDefinitions

TensorTest::TensorTest() : data(0) {
  WfeUpdateTimestamp
  setLabel("TensorTest");

  Changed.connect(capputils::EventHandler<TensorTest>(this, &TensorTest::changedHandler));
}

TensorTest::~TensorTest() {
  if (data)
    delete data;
}

void TensorTest::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void TensorTest::writeResults() {
  if (!data)
    return;

}

}

}
