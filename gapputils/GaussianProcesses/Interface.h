#pragma once
#ifndef _GAUSSIANPROCESSES_INTERFACE_H_
#define _GAUSSIANPROCESSES_INTERFACE_H_

#include <WorkflowElement.h>

namespace GaussianProcesses {

class Interface : public gapputils::workflow::WorkflowElement
{
  InitReflectableClass(Interface)

  Property(FirstColumn, int)
  Property(LastColumn, int)
  Property(GoalColumn, int)
  Property(FirstTrainRow, int)
  Property(LastTrainRow, int)
  Property(Train, std::string)
  Property(FirstTestRow, int)
  Property(LastTestRow, int)
  Property(Test, std::string)
  Property(Error, double)

public:
  Interface(void);
  virtual ~Interface(void);

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const { }
  virtual void writeResults() { }
};

}

#endif
