/*
 * LogFilter.h
 *
 *  Created on: May 17, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_MLLOGFILTER_H_
#define GAPPUTLIS_MLLOGFILTER_H_

#include <gapputils/WorkflowElement.h>

#include <culib/ICudaImage.h>
#include <capputils/Enumerators.h>

namespace gapputils {

namespace ml {

ReflectableEnum(FilterFunction, Log, Sqrt, Bernstein);

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

class FunctionFilter : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(FunctionFilter)

  Property(InputImage, boost::shared_ptr<culib::ICudaImage>)
  Property(Function, FilterFunction)
  Property(Parameters, boost::shared_ptr<FunctionParameters>)
  Property(OutputImage, boost::shared_ptr<culib::ICudaImage>)

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

}

}

#endif /* GAPPUTLIS_MLLOGFILTER_H_ */
