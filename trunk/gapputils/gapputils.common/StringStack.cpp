/*
 * StringStack.cpp
 *
 *  Created on: Jan 30, 2012
 *      Author: tombr
 */

#include "StringStack.h"

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

#include <algorithm>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(StringStack)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputVector1, Input("In1"), Observe(PROPERTY_ID))
  DefineProperty(InputVector2, Input("In2"), Observe(PROPERTY_ID))
  DefineProperty(OutputVector, Output("Out"), Observe(PROPERTY_ID))

EndPropertyDefinitions

StringStack::StringStack() : data(0) {
  WfeUpdateTimestamp
  setLabel("StringStack");

  Changed.connect(capputils::EventHandler<StringStack>(this, &StringStack::changedHandler));
}

StringStack::~StringStack() {
  if (data)
    delete data;
}

void StringStack::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void StringStack::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new StringStack();

  if (!capputils::Verifier::Valid(*this))
    return;

  const std::vector<std::string>& in1 = getInputVector1();
  const std::vector<std::string>& in2 = getInputVector2();

  std::vector<std::string> out(in1.size() + in2.size());
  std::copy(in1.begin(), in1.end(), out.begin());
  std::copy(in2.begin(), in2.end(), out.begin() + in1.size());

  data->setOutputVector(out);
}

void StringStack::writeResults() {
  if (!data)
    return;

  setOutputVector(data->getOutputVector());
}

}

}
