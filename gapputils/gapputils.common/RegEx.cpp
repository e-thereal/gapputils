/*
 * RegEx.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: tombr
 */

#include "RegEx.h"

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
#include <capputils/HideAttribute.h>

#include <gapputils/ReadOnlyAttribute.h>

#include <boost/regex.hpp>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

int RegEx::inputId;
int RegEx::regexId;
int RegEx::formatId;

BeginPropertyDefinitions(RegEx)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(Input, Input(""), Observe(inputId = Id), NotEqual<std::string>(""), TimeStamp(Id))
  DefineProperty(Output, Output(""), Observe(Id), TimeStamp(Id))
  DefineProperty(Regex, Observe(regexId = Id), NotEqual<std::string>(""), TimeStamp(Id))
  DefineProperty(Format, Observe(formatId = Id), TimeStamp(Id))

EndPropertyDefinitions

RegEx::RegEx() {
  WfeUpdateTimestamp
  setLabel("RegEx");

  Changed.connect(capputils::EventHandler<RegEx>(this, &RegEx::changedHandler));
}

RegEx::~RegEx() {
}

void RegEx::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (!capputils::Verifier::Valid(*this))
    return;

  if (eventId == inputId || eventId == regexId || eventId == formatId) {
    boost::smatch res;
    boost::regex rx(getRegex());
    if (boost::regex_search(getInput(), res, rx))
      setOutput(res.format(getFormat()));
    else
      setOutput("No match.");
  }
}

void RegEx::execute(gapputils::workflow::IProgressMonitor* monitor) const {
}

void RegEx::writeResults() {
}

}

}
