/*
 * Filename.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#include "Filename.h"

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

#include <gapputils/ReadOnlyAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {

namespace parameters {

int Filename::patternId;

BeginPropertyDefinitions(Filename, Interface())

  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Filename>)
  DefineProperty(Value, capputils::attributes::Filename(), FileExists(), Observe(Id))
  DefineProperty(Pattern, Observe(patternId = Id))

EndPropertyDefinitions

Filename::Filename() {
  setLabel("Filename");
  Changed.connect(capputils::EventHandler<Filename>(this, &Filename::changedHandler));
}

Filename::~Filename() { }

void Filename::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == patternId) {
    capputils::reflection::IClassProperty* prop = findProperty("Value");
    assert(prop);
    FilenameAttribute* attr = prop->getAttribute<FilenameAttribute>();
    assert(attr);
    attr->setPattern(getPattern());
  }
}

}

namespace inputs {

int Filename::patternId;

BeginPropertyDefinitions(Filename, Interface())

  ReflectableBase(gapputils::workflow::DefaultWorkflowElement<Filename>)
  DefineProperty(Value, Output(""), capputils::attributes::Filename(), FileExists(), Observe(Id))
  DefineProperty(Pattern, Observe(patternId = Id))

EndPropertyDefinitions

Filename::Filename() {
  setLabel("Filename");
  Changed.connect(capputils::EventHandler<Filename>(this, &Filename::changedHandler));
}

Filename::~Filename() { }

void Filename::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == patternId) {
    capputils::reflection::IClassProperty* prop = findProperty("Value");
    assert(prop);
    FilenameAttribute* attr = prop->getAttribute<FilenameAttribute>();
    assert(attr);
    attr->setPattern(getPattern());
  }
}

}

}
