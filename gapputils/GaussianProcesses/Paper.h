#pragma once

#ifndef _GAPPUTILS_PAPER_H_
#define _GAPPUTILS_PAPER_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>
#include <Enumerators.h>

namespace gapputils {

ReflectableEnum(Algorithm, GpuNaive, GpuFast, Cpu);
ReflectableEnum(Test, Prediction, Training);

class Paper : public capputils::reflection::ReflectableClass,
              public capputils::ObservableClass
{

  InitReflectableClass(Paper)

  Property(Algorithm, Algorithm)
  Property(Test, Test)
  Property(Testfile, std::string)
  Property(IterationsCount, int)
  Property(SampleCount, int)
  Property(FeatureCount, int)
  Property(TrainTime, double)
  Property(PredictionTime, double)
  Property(FirstColumn, int)
  Property(LastColumn, int)
  Property(YColumn, int)
  Property(FirstRow, int)
  Property(LastRow, int)
  Property(Run, bool)
  Property(ConfigurationName, std::string)
  Property(Result, double)

public:
  Paper(void);
  virtual ~Paper(void);

  void changeEventHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif

