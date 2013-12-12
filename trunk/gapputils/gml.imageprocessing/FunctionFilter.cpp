/*
 * FunctionFilter.cpp
 *
 *  Created on: Jan 20, 2013
 *      Author: tombr
 */

#include "FunctionFilter.h"

#include <capputils/EventHandler.h>

#include <tbblas/tensor.hpp>
#include <tbblas/math.hpp>

#include <algorithm>

namespace gml {

namespace imageprocessing {

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

GammaParameters::GammaParameters() : _Slope(1.f), _Gamma(1.f), _Intercept(0.f) { }

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

  ReflectableBase(DefaultWorkflowElement<FunctionFilter>)
  WorkflowProperty(InputImage, Input(""), NotNull<Type>())
  WorkflowProperty(Function, Enumerator<Type>(), Dummy(functionId = Id))
  WorkflowProperty(Parameters, Reflectable<Type>())
  WorkflowProperty(OutputImage, Output(""))

EndPropertyDefinitions

FunctionFilter::FunctionFilter() : _Parameters(new NoParameters()) {
  setLabel("Func");

  Changed.connect(EventHandler<FunctionFilter>(this, &FunctionFilter::changedHandler));
}

void FunctionFilter::changedHandler(capputils::ObservableClass* sender, int eventId) {
  if (eventId == functionId) {
    switch (getFunction()) {
    case FilterFunction::Abs:
    case FilterFunction::Log:
    case FilterFunction::Sqrt:
      if (!boost::dynamic_pointer_cast<NoParameters>(getParameters()))
        setParameters(boost::make_shared<NoParameters>());
      break;

    case FilterFunction::Bernstein:
      if (!boost::dynamic_pointer_cast<BernsteinParameters>(getParameters()))
        setParameters(boost::make_shared<BernsteinParameters>());
      break;

    case FilterFunction::Gamma:
      if (!boost::dynamic_pointer_cast<GammaParameters>(getParameters()))
        setParameters(boost::make_shared<GammaParameters>());
      break;

    case FilterFunction::Sigmoid:
      if (!boost::dynamic_pointer_cast<SigmoidParameters>(getParameters()))
        setParameters(boost::make_shared<SigmoidParameters>());
      break;

    case FilterFunction::Threshold:
      if (!boost::dynamic_pointer_cast<ThresholdParameters>(getParameters()))
        setParameters(boost::make_shared<ThresholdParameters>());
      break;
    }
  }
}

struct cpu_gamma {
  float slope, gamma, intercept;

  cpu_gamma(const float& slope, const float& gamma, const float& intercept) : slope(slope), gamma(gamma), intercept(intercept) { }

  float operator()(const float& x) const {
    return slope * powf(x, gamma) + intercept;
  }
};

struct cpu_sigmoid {
  float slope, inflection, minimum, invContrast;

  cpu_sigmoid(const float& slope, const float& inflection) : slope(slope), inflection(inflection) {
    minimum = 1.f / (1.f + expf(slope * inflection));
    const float maximum = 1.f / (1.f + expf(-slope * (1.f - inflection)));
    invContrast = 1.f / (maximum - minimum);
  }

  float operator()(const float& x) const {
    return (1.f / (1.f + expf(-slope * (x - inflection))) - minimum) * invContrast;
  }
};

void FunctionFilter::update(IProgressMonitor* monitor) const {
  using namespace tbblas;

  Logbook& dlog = getLogbook();

  image_t& input = *getInputImage();
  boost::shared_ptr<image_t> output(new image_t(input.getSize(), input.getPixelSize()));

  tensor<float, 3> img(input.getSize()[0], input.getSize()[1], input.getSize()[2]), img2;
  std::copy(input.begin(), input.end(), img.begin());

  switch(getFunction()) {
  case FilterFunction::Abs:
    img = sqrt(img * img);
    break;

  case FilterFunction::Log:
    img = log(img);
    break;

  case FilterFunction::Sqrt:
    img = sqrt(img);
    break;

  case FilterFunction::Bernstein: {
    BernsteinParameters* params = dynamic_cast<BernsteinParameters*>(getParameters().get());
    if (params)
      img = bernstein(img, params->getIndex(), params->getDegree());
    } break;

  case FilterFunction::Gamma: {
    GammaParameters* params = dynamic_cast<GammaParameters*>(getParameters().get());
    if (params)
      img = params->getSlope() * pow(img, params->getGamma()) + params->getIntercept();
    } break;

  /*case FilterFunction::Sigmoid: {
    SigmoidParameters* params = dynamic_cast<SigmoidParameters*>(getParameters().get());
    if (params) {
      thrust::transform(data.begin(), data.end(), data.begin(),
        gpu_sigmoid(params->getSlope(), params->getInflection()));
    }
    } break;*/

  case FilterFunction::Threshold: {
    ThresholdParameters* params = dynamic_cast<ThresholdParameters*>(getParameters().get());
    if (params)
      img = img > params->getThreshold();
    } break;

  default:
    dlog(Severity::Warning) << "Unsupported filter function: " << getFunction();
  }

  std::copy(img.begin(), img.end(), output->getData());
  newState->setOutputImage(output);
}

}

}
