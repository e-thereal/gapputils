/*
 * FunctionFilter.cpp
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#include "FunctionFilter.h"

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

DefineEnum(FilterFunction)

BeginPropertyDefinitions(FunctionParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(NoParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(BernsteinParameters)

  DefineProperty(Index, Observe(PROPERTY_ID))
  DefineProperty(Degree, Observe(PROPERTY_ID))

EndPropertyDefinitions

BernsteinParameters::BernsteinParameters() : _Index(0), _Degree(0) { }

BernsteinParameters::~BernsteinParameters() { }

int FunctionFilter::functionId;

BeginPropertyDefinitions(FunctionFilter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), ReadOnly(), Volatile(), Observe(PROPERTY_ID))
  ReflectableProperty(Function, Observe(functionId = PROPERTY_ID))
  ReflectableProperty(Parameters, Observe(PROPERTY_ID))
  DefineProperty(OutputImage, Output(""), ReadOnly(), Volatile(), Observe(PROPERTY_ID))

EndPropertyDefinitions

FunctionFilter::FunctionFilter() : _Parameters(new NoParameters()), data(0) {
  WfeUpdateTimestamp
  setLabel("Log");

  Changed.connect(capputils::EventHandler<FunctionFilter>(this, &FunctionFilter::changedHandler));
}

FunctionFilter::~FunctionFilter() {
  if (data)
    delete data;
}

void FunctionFilter::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == functionId) {
    switch (getFunction()) {
    case FilterFunction::Log:
    case FilterFunction::Sqrt:
      setParameters(boost::shared_ptr<FunctionParameters>(new NoParameters()));
      break;

    case FilterFunction::Bernstein:
      setParameters(boost::shared_ptr<FunctionParameters>(new BernsteinParameters()));
      break;
    }
  }
}

void FunctionFilter::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

}

}
