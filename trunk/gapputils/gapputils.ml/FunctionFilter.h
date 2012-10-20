/*
 * LogFilter.h
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_MLLOGFILTER_H_
#define GAPPUTLIS_MLLOGFILTER_H_

#include <gapputils/WorkflowElement.h>

#include <gapputils/Image.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace ml {

CapputilsEnumerator(FilterFunction, Log, Sqrt, Bernstein, Gamma, Sigmoid, Threshold);

class FunctionParameters : public capputils::reflection::ReflectableClass,
                           public capputils::ObservableClass
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
  virtual ~BernsteinParameters();
};

class GammaParameters : public FunctionParameters {
  InitReflectableClass(GammaParameters)

  Property(Slope, float)
  Property(Gamma, float)
  Property(Intercept, float)

public:
  GammaParameters();
  virtual ~GammaParameters();
};

class SigmoidParameters : public FunctionParameters {
  InitReflectableClass(SigmoidParameters)

  Property(Slope, float)
  Property(Inflection, float)

public:
  SigmoidParameters();
};

class ThresholdParameters : public FunctionParameters {
  InitReflectableClass(ThresholdParameters)

  Property(Threshold, float)

public:
  ThresholdParameters();
};

class FunctionFilter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FunctionFilter)

  Property(InputImage, boost::shared_ptr<image_t>)
  Property(Function, FilterFunction)
  Property(Parameters, boost::shared_ptr<FunctionParameters>)
  Property(OutputImage, boost::shared_ptr<image_t>)

private:
  mutable FunctionFilter* data;
  static int functionId;

public:
  FunctionFilter();
  virtual ~FunctionFilter();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

// wrapper around the boost version
float binomial(int n, int k);

}

}

#endif /* GAPPUTLIS_MLLOGFILTER_H_ */
