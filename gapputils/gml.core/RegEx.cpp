/*
 * RegEx.cpp
 *
 *  Created on: Feb 8, 2012
 *      Author: tombr
 */

#include "RegEx.h"

#include <capputils/attributes/DummyAttribute.h>
#include <capputils/EventHandler.h>

#include <boost/regex.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/null.hpp>

namespace bio = boost::iostreams;

namespace gml {

namespace core {

int RegEx::inputId;
int RegEx::regexId;
int RegEx::formatId;

BeginPropertyDefinitions(RegEx)

  ReflectableBase(DefaultWorkflowElement<RegEx>)
  WorkflowProperty(Input, Input(""), Dummy(inputId = Id), NotEmpty<Type>())
  WorkflowProperty(Regex, Dummy(regexId = Id), NotEmpty<Type>())
  WorkflowProperty(Format, Dummy(formatId = Id), NotEmpty<Type>())
  WorkflowProperty(Output, Output(""))

EndPropertyDefinitions

RegEx::RegEx() {
  setLabel("RegEx");
  Changed.connect(EventHandler<RegEx>(this, &RegEx::changedHandler));
}

void RegEx::changedHandler(ObservableClass* sender, int eventId) {
  bio::stream<bio::null_sink> nullOstream((bio::null_sink()));
  if (!Verifier::Valid(*this, nullOstream))
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

}

}
