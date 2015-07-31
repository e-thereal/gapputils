/*
 * Filenames.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#include "Filenames.h"

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
#include <capputils/attributes/EnumerableAttribute.h>
#include <capputils/attributes/FromEnumerableAttribute.h>
#include <capputils/attributes/ToEnumerableAttribute.h>

#include <gapputils/attributes/ReadOnlyAttribute.h>
#include <gapputils/attributes/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {

namespace inputs {

int Filenames::filenamesId;
int Filenames::patternId;

BeginPropertyDefinitions(Filenames, Interface())

  ReflectableBase(gapputils::workflow::CollectionElement)
  DefineProperty(Values, Output("Files"), Filename("All (*)", true), FileExists(), Enumerable<Type, false>(), Observe(filenamesId = Id))
  DefineProperty(Value, Output("File"), Filename(), FileExists(), FromEnumerable(filenamesId), Observe(Id))
  DefineProperty(Pattern, Observe(patternId = Id))

EndPropertyDefinitions

Filenames::Filenames() {
  setLabel("Filenames");

  Changed.connect(capputils::EventHandler<Filenames>(this, &Filenames::changedHandler));
}

void Filenames::changedHandler(capputils::ObservableClass* /*sender*/, int eventId) {
  if (eventId == patternId) {
    capputils::reflection::IClassProperty* prop = findProperty("Values");
    assert(prop);
    FilenameAttribute* attr = prop->getAttribute<FilenameAttribute>();
    assert(attr);
    attr->setPattern(getPattern());
  }
}

}

namespace outputs {

int Filenames::filenamesId;

BeginPropertyDefinitions(Filenames, Interface())

  ReflectableBase(gapputils::workflow::CollectionElement)
  DefineProperty(Values, Input("Files"), Filename("All (*)", true), Enumerable<Type, false>(), Observe(filenamesId = Id))
  DefineProperty(Value, Input("File"), Filename(), FileExists(), ToEnumerable(filenamesId), Observe(Id))

EndPropertyDefinitions

Filenames::Filenames() {
  setLabel("Filenames");
}

}

}
