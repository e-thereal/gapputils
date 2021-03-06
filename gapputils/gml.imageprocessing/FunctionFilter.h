/*
 * FunctionFilter.h
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#ifndef GML_FUNCTIONFILTER_H_
#define GML_FUNCTIONFILTER_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/Image.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace imageprocessing {

CapputilsEnumerator(FilterFunction, Abs, Log, Sqrt, Bernstein, Gamma, Sigmoid, Threshold);

class FunctionParameters : public capputils::reflection::ReflectableClass,
                           public ObservableClass
{
  InitReflectableClass(FunctionParameters)
};

class NoParameters : public FunctionParameters {
  InitReflectableClass(NoParameters)
};

class BernsteinParameters : public FunctionParameters {
  InitReflectableClass(BernsteinParameters)

  Property(Index, int)
  Property(Degree, int)

public:
  BernsteinParameters();
};

class GammaParameters : public FunctionParameters {
  InitReflectableClass(GammaParameters)

  Property(Slope, double)
  Property(Gamma, double)
  Property(Intercept, double)

public:
  GammaParameters();
};

class SigmoidParameters : public FunctionParameters {
  InitReflectableClass(SigmoidParameters)

  Property(Slope, double)
  Property(Inflection, double)

public:
  SigmoidParameters();
};

class ThresholdParameters : public FunctionParameters {
  InitReflectableClass(ThresholdParameters)

  Property(Threshold, double)

public:
  ThresholdParameters();
};

class FunctionFilter : public DefaultWorkflowElement<FunctionFilter> {

  InitReflectableClass(FunctionFilter)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Function, FilterFunction)
  Property(Parameters, boost::shared_ptr<FunctionParameters>)
  Property(OutputImage, boost::shared_ptr<image_t>)

  static int functionId;

public:
  FunctionFilter();

protected:
  virtual void update(IProgressMonitor* monitor) const;

  void changedHandler(ObservableClass* sender, int eventId);
};

}

}

#endif /* GML_FUNCTIONFILTER_H_ */
