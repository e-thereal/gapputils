#pragma once

#ifndef _GAPPUTILS_GPREGRESSION_H_
#define _GAPPUTILS_GPREGRESSION_H_

#include <WorkflowElement.h>

namespace GaussianProcesses {

class GPRegression : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(GPRegression)

  Property(Label, std::string)
  Property(FeatureCount, int)
  Property(X, double*)
  Property(Y, double*)
  Property(TrainingCount, int)

  Property(Xstar, double*)
  Property(Ystar, double*)
  Property(TestCount, int)

  Property(Regress, bool)

public:
  GPRegression(void);
  virtual ~GPRegression(void);

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif
