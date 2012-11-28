/*
 * Filenames.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#include "Filenames.h"

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
#include <capputils/EnumerableAttribute.h>
#include <capputils/FromEnumerableAttribute.h>
#include <capputils/ToEnumerableAttribute.h>

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace interfaces {

namespace inputs {

int Filenames::filenamesId;

BeginPropertyDefinitions(Filenames, Interface())

  ReflectableBase(gapputils::workflow::CollectionElement)
  DefineProperty(Values, Output("Files"), Filename("All (*)", true), FileExists(), Enumerable<TYPE_OF(Values), false>(), Observe(filenamesId = Id))
  DefineProperty(Value, Output("File"), Filename(), FileExists(), FromEnumerable(filenamesId), Observe(Id))

EndPropertyDefinitions

Filenames::Filenames() {
  setLabel("Filenames");
}

}

namespace outputs {

int Filenames::filenamesId;

BeginPropertyDefinitions(Filenames, Interface())

  ReflectableBase(gapputils::workflow::CollectionElement)
  DefineProperty(Values, Input("Files"), Filename("All (*)", true), Enumerable<TYPE_OF(Values), false>(), Observe(filenamesId = Id))
  DefineProperty(Value, Input("File"), Filename(), FileExists(), ToEnumerable(filenamesId), Observe(Id))

EndPropertyDefinitions

Filenames::Filenames() {
  setLabel("Filenames");
}

}

}
