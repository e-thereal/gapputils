/*
 * Function.cpp
 *
 *  Created on: Jan 31, 2013
 *      Author: tombr
 */

#include "Function.h"

#include <capputils/EventHandler.h>

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

#include <algorithm>

namespace gml {

namespace core {

BeginPropertyDefinitions(FunctionParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(NoParameters)
EndPropertyDefinitions

BeginPropertyDefinitions(BernsteinParameters)

  DefineProperty(Index, Observe(Id))
  DefineProperty(Degree, Observe(Id))

EndPropertyDefinitions

BernsteinParameters::BernsteinParameters() : _Index(0), _Degree(0) { }

BeginPropertyDefinitions(GammaParameters)
  DefineProperty(Slope, Observe(Id))
  DefineProperty(Gamma, Observe(Id))
  DefineProperty(Intercept, Observe(Id))
EndPropertyDefinitions

GammaParameters::GammaParameters() : _Slope(1), _Gamma(1), _Intercept(0) { }

BeginPropertyDefinitions(SigmoidParameters)
  DefineProperty(Slope)
  DefineProperty(Inflection)
EndPropertyDefinitions

SigmoidParameters::SigmoidParameters() : _Slope(1), _Inflection(0.5) { }

BeginPropertyDefinitions(ThresholdParameters)
  DefineProperty(Threshold)
EndPropertyDefinitions

ThresholdParameters::ThresholdParameters() : _Threshold(0) { }

BeginPropertyDefinitions(ClippingParameters)
  DefineProperty(Minimum)
  DefineProperty(Maximum)
EndPropertyDefinitions

ClippingParameters::ClippingParameters() : _Minimum(0), _Maximum(1) { }

int Function::functionId;

BeginPropertyDefinitions(Function)

  ReflectableBase(DefaultWorkflowElement<Function>)
  WorkflowProperty(Inputs, Input(""), NotNull<Type>())
  WorkflowProperty(Function, Enumerator<Type>(), Dummy(functionId = Id))
  WorkflowProperty(Parameters, Reflectable<Type>())
  WorkflowProperty(Outputs, Output(""))

EndPropertyDefinitions

Function::Function() : _Parameters(new NoParameters()) {
  setLabel("Log");

  Changed.connect(EventHandler<Function>(this, &Function::changedHandler));
}

void Function::changedHandler(ObservableClass* sender, int eventId) {
  if (eventId == functionId) {
    switch (getFunction()) {
    case Functions::Abs:
    case Functions::Log:
    case Functions::Sqrt:
      if (!boost::dynamic_pointer_cast<NoParameters>(getParameters()))
        setParameters(boost::make_shared<NoParameters>());
      break;

    case Functions::Bernstein:
      if (!boost::dynamic_pointer_cast<BernsteinParameters>(getParameters()))
        setParameters(boost::make_shared<BernsteinParameters>());
      break;

    case Functions::Gamma:
      if (!boost::dynamic_pointer_cast<GammaParameters>(getParameters()))
        setParameters(boost::make_shared<GammaParameters>());
      break;

    case Functions::Sigmoid:
      if (!boost::dynamic_pointer_cast<SigmoidParameters>(getParameters()))
        setParameters(boost::make_shared<SigmoidParameters>());
      break;

    case Functions::Threshold:
      if (!boost::dynamic_pointer_cast<ThresholdParameters>(getParameters()))
        setParameters(boost::make_shared<ThresholdParameters>());
      break;

    case Functions::Clipping:
      if (!boost::dynamic_pointer_cast<ClippingParameters>(getParameters()))
        setParameters(boost::make_shared<ClippingParameters>());
      break;
    }
  }
}

struct cpu_sigmoid {
  double slope, inflection, minimum, invContrast;

  cpu_sigmoid(const double& slope, const double& inflection) : slope(slope), inflection(inflection) {
    minimum = 1.0 / (1.0 + exp(slope * inflection));
    const double maximum = 1.0 / (1.0 + exp(-slope * (1.0 - inflection)));
    invContrast = 1.0 / (maximum - minimum);
  }

  double operator()(const double& x) const {
    return (1.0 / (1.0 + exp(-slope * (x - inflection))) - minimum) * invContrast;
  }
};

void Function::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  std::vector<double>& inputs = *getInputs();
  boost::shared_ptr<std::vector<double> > outputs(new std::vector<double>(inputs.size()));

  tensor<double, 1> data(inputs.size());
  std::copy(inputs.begin(), inputs.end(), data.begin());

  switch(getFunction()) {
  case Functions::Abs:
    data = abs(data);
    break;

  case Functions::Log:
    data = log(data);
    break;

  case Functions::Sqrt:
    data = sqrt(data);
    break;

  case Functions::Bernstein: {
    BernsteinParameters* params = dynamic_cast<BernsteinParameters*>(getParameters().get());
    if (params)
      data = bernstein(data, params->getIndex(), params->getDegree());
    } break;

  case Functions::Gamma: {
    GammaParameters* params = dynamic_cast<GammaParameters*>(getParameters().get());
    if (params)
      data = params->getSlope() * pow(data, params->getGamma()) + params->getIntercept();
    } break;

  /*case FilterFunction::Sigmoid: {
    SigmoidParameters* params = dynamic_cast<SigmoidParameters*>(getParameters().get());
    if (params) {
      thrust::transform(data.begin(), data.end(), data.begin(),
        gpu_sigmoid(params->getSlope(), params->getInflection()));
    }
    } break;*/

  case Functions::Threshold: {
    ThresholdParameters* params = dynamic_cast<ThresholdParameters*>(getParameters().get());
    if (params)
      data = data > params->getThreshold();
    } break;

  case Functions::Clipping: {
    ClippingParameters* params = dynamic_cast<ClippingParameters*>(getParameters().get());
    if (params)
      data = max(params->getMinimum(), min(params->getMaximum(), data));
    } break;

  default:
    dlog(Severity::Warning) << "Unsupported function: " << getFunction();
  }

  std::copy(data.begin(), data.end(), outputs->begin());
  newState->setOutputs(outputs);
}

}

}