/*
 * StringReplacer.cpp
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#include "StringReplacer.h"

#include <capputils/NotEmptyAttribute.h>
#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/Verifier.h>
#include <capputils/SerializeAttribute.h>
#include <capputils/TimeStampAttribute.h>

#include <gapputils/CacheableAttribute.h>

#include <capputils/Logbook.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

int StringReplacer::inputId;
int StringReplacer::findId;
int StringReplacer::replaceId;

BeginPropertyDefinitions(StringReplacer)

  ReflectableBase(workflow::WorkflowElement)
  DefineProperty(Input, Input("In"), Observe(inputId = Id), NotEmpty<Type>(), TimeStamp(Id))
  DefineProperty(Output, Output("Out"), Observe(Id), TimeStamp(Id))
  DefineProperty(Find, Observe(findId = Id), NotEmpty<Type>(), TimeStamp(Id))
  DefineProperty(Replace, Observe(replaceId = Id), TimeStamp(Id))

EndPropertyDefinitions

string replaceAll(const string& context, const string& from, const string& to)
{
  string str = context;
  size_t lookHere = 0;
  size_t foundHere;
  while((foundHere = str.find(from, lookHere)) != string::npos)
  {
    str.replace(foundHere, from.size(), to);
        lookHere = foundHere + to.size();
  }
  return str;
}

StringReplacer::StringReplacer() {
  setLabel("Replace");
  Changed.connect(capputils::EventHandler<StringReplacer>(this, &StringReplacer::changedHandler));
}

StringReplacer::~StringReplacer() {
}

void StringReplacer::changedHandler(capputils::ObservableClass* sender, int eventId) {
  capputils::Logbook& dlog = getLogbook();

  if (!capputils::Verifier::Valid(*this, dlog))
    return;

  if (eventId == inputId || eventId == findId || eventId == replaceId) {
    setOutput(replaceAll(getInput(), getFind(), getReplace()));
  }
}

void StringReplacer::writeResults() {
  setOutput(getOutput());
}

}

}
