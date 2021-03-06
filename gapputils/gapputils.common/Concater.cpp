/*
 * Concat.cpp
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#include "Concater.h"

#include <capputils/ObserveAttribute.h>
#include <capputils/EventHandler.h>
#include <capputils/InputAttribute.h>
#include <capputils/OutputAttribute.h>
#include <capputils/NotEqualAssertion.h>
#include <capputils/Verifier.h>
#include <capputils/TimeStampAttribute.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

namespace common {

int Concater::outputId;

BeginPropertyDefinitions(Concater)

  ReflectableBase(workflow::WorkflowElement)
  DefineProperty(Input1, Input("In1"), Observe(Id), TimeStamp(Id))
  DefineProperty(Input2, Input("In2"), Observe(Id), TimeStamp(Id))
  DefineProperty(Output, Output("Out"), Observe(outputId = Id), TimeStamp(Id))
  DefineProperty(Separator, Observe(Id), TimeStamp(Id))

EndPropertyDefinitions

Concater::Concater() : _Separator(" ") {
  setLabel("Concat");
  Changed.connect(capputils::EventHandler<Concater>(this, &Concater::changedHandler));
}

Concater::~Concater() {
}

void Concater::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == outputId)
    return;
  setOutput(getInput1() + getSeparator() + getInput2());
}

}

}
