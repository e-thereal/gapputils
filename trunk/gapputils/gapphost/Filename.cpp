/*
 * Filename.cpp
 *
 *  Created on: Jun 7, 2012
 *      Author: tombr
 */

#include "Filename.h"

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

#include <gapputils/ReadOnlyAttribute.h>
#include <gapputils/InterfaceAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace host {

namespace inputs {

BeginPropertyDefinitions(Filename, Interface(), Deprecated("Use 'interfaces::parameters::Filename' instead."))

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Value, Output(""), capputils::attributes::Filename(), FileExists(), Observe(Id))

EndPropertyDefinitions

Filename::Filename() : data(0) {
  WfeUpdateTimestamp
  setLabel("Filename");
}

Filename::~Filename() {
  if (data)
    delete data;
}

void Filename::execute(gapputils::workflow::IProgressMonitor* /*monitor*/) const {
  if (!data)
    data = new Filename();

  if (!capputils::Verifier::Valid(*this))
    return;


}

void Filename::writeResults() {
  if (!data)
    return;

}

}

}

}
