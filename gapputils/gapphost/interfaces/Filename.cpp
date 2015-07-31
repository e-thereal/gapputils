/*
 * Filename.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#include "Filename.h"

#include <capputils/EventHandler.h>
#include <capputils/Verifier.h>
#include <capputils/attributes/FileExistsAttribute.h>
#include <capputils/attributes/FilenameAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/NotEqualAttribute.h>
#include <capputils/attributes/ObserveAttribute.h>
#include <capputils/attributes/OutputAttribute.h>
#include <capputils/attributes/TimeStampAttribute.h>
#include <capputils/attributes/VolatileAttribute.h>

#include <gapputils/attributes/ReadOnlyAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

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

void Filename::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
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

void Filename::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
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
