#pragma once
#ifndef _GAUSSIANPROCESSES_INTERFACE_H_
#define _GAUSSIANPROCESSES_INTERFACE_H_

#include <gapputils/WorkflowInterface.h>

namespace GaussianProcesses {

class Interface : public gapputils::workflow::WorkflowInterface
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
};

}

#endif
