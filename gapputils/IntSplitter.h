#pragma once
#ifndef _GAUSSIANPROCESSES_INTSPLITTER_H_
#define _GAUSSIANPROCESSES_INTSPLITTER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace GaussianProcesses {

class IntSplitter : public capputils::reflection::ReflectableClass,
                    public capputils::ObservableClass
{
  InitReflectableClass(IntSplitter)

  Property(Label, std::string)
  Property(In, int)
  Property(Out1, int)
  Property(Out2, int)

public:
  IntSplitter(void);
  virtual ~IntSplitter(void);
};

class IntSplitter4 : public capputils::reflection::ReflectableClass,
                    public capputils::ObservableClass
{
  InitReflectableClass(IntSplitter4)

  Property(Label, std::string)
  Property(In, int)
  Property(Out1, int)
  Property(Out2, int)
  Property(Out3, int)
  Property(Out4, int)

public:
  IntSplitter4(void);
  virtual ~IntSplitter4(void);
};

}

#endif
