/*
 * OnOfN.cpp
 *
 *  Created on: Oct 25, 2011
 *      Author: tombr
 */

#include "OneOfN.h"

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

#include <gapputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(OneOfN)

  ReflectableBase(gapputils::workflow::WorkflowElement)

  DefineProperty(OnIndex, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Count, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Vector, Output("Vec"), Volatile(), ReadOnly(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

OneOfN::OneOfN() : _OnIndex(0), _Count(1), data(0) {
  WfeUpdateTimestamp
  setLabel("OneOfN");

  Changed.connect(capputils::EventHandler<OneOfN>(this, &OneOfN::changedHandler));
}

OneOfN::~OneOfN() {
  if (data)
    delete data;
}

void OneOfN::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void OneOfN::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new OneOfN();

  if (!capputils::Verifier::Valid(*this))
    return;

  boost::shared_ptr<std::vector<float> > vec(new std::vector<float>(getCount(), 0.f));
  vec->at(getOnIndex()) = 1.f;

  data->setVector(vec);
}

void OneOfN::writeResults() {
  if (!data)
    return;

  setVector(data->getVector());
}

}

}
