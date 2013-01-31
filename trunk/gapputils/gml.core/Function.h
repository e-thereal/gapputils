/*
 * FunctionFilter.h
 *
 *  Created on: Jan 31, 2013
 *      Author: tombr
 */

#ifndef GML_FUNCTION_H_
#define GML_FUNCTION_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

#include <capputils/Enumerators.h>

namespace gml {

namespace core {

CapputilsEnumerator(Functions, Abs, Log, Sqrt, Bernstein, Gamma, Sigmoid, Threshold, Clipping);

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

class ClippingParameters : public FunctionParameters {
  InitReflectableClass(ClippingParameters)

  Property(Minimum, double)
  Property(Maximum, double)

public:
  ClippingParameters();
};

class Function : public DefaultWorkflowElement<Function> {

  InitReflectableClass(Function)

  Property(Inputs, boost::shared_ptr<std::vector<double> >)
  Property(Function, Functions)
  Property(Parameters, boost::shared_ptr<FunctionParameters>)
  Property(Outputs, boost::shared_ptr<std::vector<double> >)

  static int functionId;

public:
  Function();

protected:
  virtual void update(IProgressMonitor* monitor) const;

  void changedHandler(ObservableClass* sender, int eventId);
};

}

}

#endif /* GML_FUNCTION_H_ */
