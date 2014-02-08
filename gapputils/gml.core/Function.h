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

#include <tbblas/tensor.hpp>

namespace gml {

namespace core {

CapputilsEnumerator(Functions, Abs, Exp, Log, Sqrt, Bernstein, Gamma, Sigmoid, Threshold, Clipping, Axpb);

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

class AxpbParameters : public FunctionParameters {
  InitReflectableClass(AxpbParameters)

  Property(Slope, double)
  Property(Intercept, double)

public:
  AxpbParameters();
};

class Function : public DefaultWorkflowElement<Function> {

  typedef std::vector<double> data_t;
  typedef std::vector<boost::shared_ptr<data_t> > v_data_t;

  InitReflectableClass(Function)

  Property(Input, boost::shared_ptr<data_t>)
  Property(Inputs, boost::shared_ptr<v_data_t>)
  Property(Function, Functions)
  Property(Parameters, boost::shared_ptr<FunctionParameters>)
  Property(Output, boost::shared_ptr<data_t>)
  Property(Outputs, boost::shared_ptr<v_data_t>)

  static int functionId;

public:
  Function();

protected:
  virtual void update(IProgressMonitor* monitor) const;

  void changedHandler(ObservableClass* sender, int eventId);

protected:
  void convertData(tbblas::tensor<double, 1>& data) const;
};

}

}

#endif /* GML_FUNCTION_H_ */
