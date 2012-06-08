/*
 * GenerateVector.cpp
 *
 *  Created on: Aug 4, 2011
 *      Author: tombr
 */

#include "GenerateVector.h"

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

#include <algorithm>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace common {

BeginPropertyDefinitions(GenerateVector)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(From, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(Step, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(To, Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))
  DefineProperty(OutputVector, Output("PV"), ReadOnly(), Volatile(), Observe(PROPERTY_ID), TimeStamp(PROPERTY_ID))

EndPropertyDefinitions

GenerateVector::GenerateVector() : _From(0.f), _Step(0.1), _To(1.0), data(0) {
  WfeUpdateTimestamp
  setLabel("Gen");

  Changed.connect(capputils::EventHandler<GenerateVector>(this, &GenerateVector::changedHandler));
}

GenerateVector::~GenerateVector() {
  if (data)
    delete data;
}

void GenerateVector::changedHandler(capputils::ObservableClass* sender, int eventId) {

}

void GenerateVector::execute(gapputils::workflow::IProgressMonitor* monitor) const {
  if (!data)
    data = new GenerateVector();

  if (!capputils::Verifier::Valid(*this))
    return;

  const float from = getFrom(), to = getTo(), step = getStep();
  const int count = (to - from) / step + 1;

  boost::shared_ptr<std::vector<float> > output(new std::vector<float>(count));
  for (int i = 0; i < count; ++i) {
    output->at(i) = (float)i * step + from;
  }
  data->setOutputVector(output);
}

void GenerateVector::writeResults() {
  if (!data)
    return;

  setOutputVector(data->getOutputVector());
}

}

}
