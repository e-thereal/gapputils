#pragma once

#ifndef gapputils_GP_H
#define gapputils_GP_H

#include <WorkflowElement.h>

#include <vector>

namespace GaussianProcesses {

class GPTest : public gapputils::workflow::WorkflowElement
{

  InitReflectableClass(GPTest)

  Property(Label, std::string)
  Property(X, std::vector<float>)
  Property(Y, std::vector<float>)
  Property(OutputName, std::string)
  Property(First, float)
  Property(Step, float)
  Property(Last, float)
  Property(Xstar, std::vector<float>)
  Property(Mu, std::vector<float>)
  Property(CI, std::vector<float>)
  Property(SigmaF, float)
  Property(Length, float)
  Property(SigmaN, float)
  Property(Train, bool)

private:
  mutable GPTest* data;

public:
  GPTest(void);
  virtual ~GPTest(void);

  void changeHandler(capputils::ObservableClass* sender, int eventId);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();
};

}

#endif
