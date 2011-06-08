#pragma once

#ifndef _GAPPUTILS_GPREGRESSION_H_
#define _GAPPUTILS_GPREGRESSION_H_

#include <gapputils/WorkflowElement.h>

namespace GaussianProcesses {

class GPRegression : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(GPRegression)

  Property(FeatureCount, int)
  Property(X, double*)
  Property(Y, double*)
  Property(TrainingCount, int)

  Property(Xstar, double*)
  Property(Ystar, double*)
  Property(StandardDeviation, double*)
  Property(TestCount, int)

private:
  mutable GPRegression* data;
  static int xId, dId;

public:
  GPRegression(void);
  virtual ~GPRegression(void);

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif
