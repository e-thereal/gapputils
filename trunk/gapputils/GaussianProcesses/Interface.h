#pragma once
#ifndef _GAUSSIANPROCESSES_INTERFACE_H_
#define _GAUSSIANPROCESSES_INTERFACE_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace GaussianProcesses {

class Interface : public capputils::reflection::ReflectableClass,
                            public capputils::ObservableClass
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

public:
  Interface(void);
  virtual ~Interface(void);
};

}

#endif
