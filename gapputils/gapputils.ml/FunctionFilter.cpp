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

#include <boost/math/special_functions/binomial.hpp>

#include <culib/CudaImage.h>
#include <iostream>

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

BeginPropertyDefinitions(GammaParameters)
  DefineProperty(Slope, Observe(PROPERTY_ID))
  DefineProperty(Gamma, Observe(PROPERTY_ID))
  DefineProperty(Intercept, Observe(PROPERTY_ID))
EndPropertyDefinitions

GammaParameters::GammaParameters() : _Slope(1.f), _Gamma(1.f), _Intercept(0.f) { }
GammaParameters::~GammaParameters() { }

BeginPropertyDefinitions(SigmoidParameters)
  DefineProperty(Slope)
  DefineProperty(Inflection)
EndPropertyDefinitions

SigmoidParameters::SigmoidParameters() : _Slope(1.f), _Inflection(0.5) { }

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

    case FilterFunction::Gamma:
      setParameters(boost::shared_ptr<GammaParameters>(new GammaParameters()));
      break;

    case FilterFunction::Sigmoid:
      setParameters(boost::shared_ptr<SigmoidParameters>(new SigmoidParameters()));
      break;
    }
  }
}

void FunctionFilter::writeResults() {
  if (!data)
    return;

  setOutputImage(data->getOutputImage());
}

float binomial(int n, int k) {
  return boost::math::binomial_coefficient<float>(n, k);
}

}

}
