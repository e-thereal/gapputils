/*
 * String.cpp
 *
 *  Created on: Jun 8, 2012
 *      Author: tombr
 */

#include "StringInterface.h"

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
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace host {

namespace inputs {

BeginPropertyDefinitions(String, Interface(), Deprecated("Use 'interfaces::inputs::String' instead."))

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Value, Output(""), Observe(Id))

EndPropertyDefinitions

String::String() : data(0) {
  WfeUpdateTimestamp
  setLabel("String");

  Changed.connect(capputils::EventHandler<String>(this, &String::changedHandler));
}

String::~String() {
  if (data)
    delete data;
}

void String::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void String::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new String();

  if (!capputils::Verifier::Valid(*this))
    return;


}

void String::writeResults() {
  if (!data)
    return;

}

}

namespace outputs {

BeginPropertyDefinitions(String, Interface(), Deprecated("Use 'interfaces::outputs::String' instead."))

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Value, Input(""), Observe(Id))

EndPropertyDefinitions

String::String() : data(0) {
  WfeUpdateTimestamp
  setLabel("String");

  Changed.connect(capputils::EventHandler<String>(this, &String::changedHandler));
}

String::~String() {
  if (data)
    delete data;
}

void String::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void String::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new String();

  if (!capputils::Verifier::Valid(*this))
    return;


}

void String::writeResults() {
  if (!data)
    return;

}

}

}

}
