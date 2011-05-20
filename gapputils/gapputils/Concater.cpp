/*
 * Concat.cpp
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#include "Concater.h"

#include <ObserveAttribute.h>
#include <EventHandler.h>
#include "InputAttribute.h"
#include "OutputAttribute.h"
#include <NotEqualAssertion.h>
#include <Verifier.h>

using namespace capputils::attributes;
using namespace std;

namespace gapputils {

int Concater::outputId;

BeginPropertyDefinitions(Concater)

  ReflectableBase(workflow::DefaultWorkflowElement)
  DefineProperty(Input1, Input("In1"), Observe(PROPERTY_ID))
  DefineProperty(Input2, Input("In2"), Observe(PROPERTY_ID))
  DefineProperty(Output, Output("Out"), Observe(outputId = PROPERTY_ID))
  DefineProperty(Separator, Observe(PROPERTY_ID))

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
