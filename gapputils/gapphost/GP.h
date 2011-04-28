#pragma once

#ifndef gapputils_GP_H
#define gapputils_GP_H

#include <ReflectableClass.h>
#include <ObservableClass.h>

#include <vector>

namespace gapputils {

class GP : public capputils::reflection::ReflectableClass,
           public capputils::ObservableClass
{

  InitReflectableClass(GP)

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
  Property(Auto, bool)
  Property(Train, bool)

public:
  GP(void);
  virtual ~GP(void);

  void changeHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif