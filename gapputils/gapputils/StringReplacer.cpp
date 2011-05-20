/*
 * StringReplacer.cpp
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#include "StringReplacer.h"

#include <ObserveAttribute.h>
#include <EventHandler.h>
#include <InputAttribute.h>
#include <OutputAttribute.h>
#include <NotEqualAssertion.h>
#include <Verifier.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

int StringReplacer::inputId;
int StringReplacer::findId;
int StringReplacer::replaceId;

BeginPropertyDefinitions(StringReplacer)

  ReflectableBase(workflow::DefaultWorkflowElement)
  DefineProperty(Input, Input("In"), Observe(inputId = PROPERTY_ID), NotEqual<string>(""))
  DefineProperty(Output, Output("Out"), Observe(PROPERTY_ID))
  DefineProperty(Find, Observe(findId = PROPERTY_ID), NotEqual<string>(""))
  DefineProperty(Replace, Observe(replaceId = PROPERTY_ID))

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
  if (!capputils::Verifier::Valid(*this))
    return;

  if (eventId == inputId || eventId == findId || eventId == replaceId) {
    setOutput(replaceAll(getInput(), getFind(), getReplace()));
  }
}

}
