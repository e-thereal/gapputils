/*
 * FunctionFilter.cpp
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#include "FunctionFilter.h"

#include <capputils/EnumeratorAttribute.h>
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

#include <capputils/HideAttribute.h>
#include <gapputils/ReadOnlyAttribute.h>

#include <boost/math/special_functions/binomial.hpp>

#include <culib/CudaImage.h>
#include <iostream>

using namespace capputils::attributes;
using namespace gapputils::attributes;

namespace gapputils {

namespace ml {

BeginPropertyDefinitions(FunctionParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(NoParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(BernsteinParameters)

  DefineProperty(Index, Observe(Id))
  DefineProperty(Degree, Observe(Id))

EndPropertyDefinitions

BernsteinParameters::BernsteinParameters() : _Index(0), _Degree(0) { }

BernsteinParameters::~BernsteinParameters() { }

BeginPropertyDefinitions(GammaParameters)
  DefineProperty(Slope, Observe(Id))
  DefineProperty(Gamma, Observe(Id))
  DefineProperty(Intercept, Observe(Id))
EndPropertyDefinitions

GammaParameters::GammaParameters() : _Slope(1.f), _Gamma(1.f), _Intercept(0.f) { }
GammaParameters::~GammaParameters() { }

BeginPropertyDefinitions(SigmoidParameters)
  DefineProperty(Slope)
  DefineProperty(Inflection)
EndPropertyDefinitions

SigmoidParameters::SigmoidParameters() : _Slope(1.f), _Inflection(0.5) { }

BeginPropertyDefinitions(ThresholdParameters)
  DefineProperty(Threshold)
EndPropertyDefinitions

ThresholdParameters::ThresholdParameters() : _Threshold(0.0f) { }

int FunctionFilter::functionId;

BeginPropertyDefinitions(FunctionFilter)

  ReflectableBase(gapputils::workflow::WorkflowElement)
  DefineProperty(InputImage, Input(""), ReadOnly(), Volatile(), Observe(Id))
  DefineProperty(Function, Enumerator<FilterFunction>(), Observe(functionId = Id))
  ReflectableProperty(Parameters, Observe(Id))
  DefineProperty(OutputImage, Output(""), ReadOnly(), Volatile(), Observe(Id))

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

    case FilterFunction::Threshold:
      setParameters(boost::shared_ptr<ThresholdParameters>(new ThresholdParameters()));
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
