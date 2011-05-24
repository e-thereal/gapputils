#pragma once

#ifndef _GAPPHOST_DATAMODEL_H_
#define _GAPPHOST_DATAMODEL_H_

#include <capputils/ReflectableClass.h>

#include "Workflow.h"

namespace gapputils {

namespace host {

class DataModel : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(DataModel)

  Property(NoGui, bool)
  Property(Help, bool)
  Property(AutoReload, bool)
  Property(WindowX, int)
  Property(WindowY, int)
  Property(WindowWidth, int)
  Property(WindowHeight, int)
  Property(MainWorkflow, workflow::Workflow*)

private:
  static DataModel* instance;

protected:
  DataModel(void);

public:
  virtual ~DataModel(void);

  static DataModel& getInstance();

  void saveToFile(const char* filename);
};

}

}

#endif
