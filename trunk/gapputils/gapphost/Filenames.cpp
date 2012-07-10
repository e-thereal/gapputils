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

namespace gapputils {

namespace host {

namespace inputs {

int Filenames::filenamesId;

BeginPropertyDefinitions(Filenames, Interface())
  using namespace capputils::attributes;

  ReflectableBase(gapputils::workflow::CollectionElement)
  DefineProperty(Values, Output("Files"), Filename("All (*)", true), FileExists(), Enumerable<TYPE_OF(Values), false>(), Observe(filenamesId = PROPERTY_ID))
  DefineProperty(Value, Output("File"), Filename(), FileExists(), FromEnumerable(filenamesId), Observe(PROPERTY_ID))

EndPropertyDefinitions

Filenames::Filenames() {
  WfeUpdateTimestamp
  setLabel("Filenames");
}

Filenames::~Filenames() {
}

void Filenames::execute(gapputils::workflow::IProgressMonitor* monitor) const {
}

void Filenames::writeResults() {
}

}

namespace outputs {

int Filenames::filenamesId;

BeginPropertyDefinitions(Filenames, Interface())
  using namespace capputils::attributes;

  ReflectableBase(gapputils::workflow::CollectionElement)
  DefineProperty(Values, Input("Files"), Filename("All (*)", true), Enumerable<TYPE_OF(Values), false>(), Observe(filenamesId = PROPERTY_ID))
  DefineProperty(Value, Input("File"), Filename(), FileExists(), ToEnumerable(filenamesId), Observe(PROPERTY_ID))

EndPropertyDefinitions

Filenames::Filenames() {
  WfeUpdateTimestamp
  setLabel("Filenames");
}

Filenames::~Filenames() {
}

void Filenames::execute(gapputils::workflow::IProgressMonitor* monitor) const {
}

void Filenames::writeResults() {
}

}

}

}
