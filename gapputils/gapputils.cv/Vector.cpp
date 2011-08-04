/*
 * Vector.cpp
 *
 *  Created on: Aug 4, 2011
 *      Author: tombr
 */

#include "Vector.h"

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

#include <algorithm>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace cv {

BeginPropertyDefinitions(Vector)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputVector, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputVector, Output("PV"), Hide(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

Vector::Vector() : data(0) {
  WfeUpdateTimestamp
  setLabel("Vector");

  Changed.connect(capputils::EventHandler<Vector>(this, &Vector::changedHandler));
}

Vector::~Vector() {
  if (data)
    delete data;
}

void Vector::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void Vector::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new Vector();

  if (!capputils::Verifier::Valid(*this))
    return;

  const std::vector<float>& input = getInputVector();
  boost::shared_ptr<std::vector<float> > output(new std::vector<float>(input.begin(), input.end()));
  data->setOutputVector(output);
}

void Vector::writeResults() {
  if (!data)
    return;

  setOutputVector(data->getOutputVector());
}

}

}
