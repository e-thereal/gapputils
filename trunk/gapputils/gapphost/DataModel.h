#pragma once

#ifndef _GAPPHOST_DATAMODEL_H_
#define _GAPPHOST_DATAMODEL_H_

#include <ReflectableClass.h>

#include "Graph.h"

namespace gapputils {

namespace host {

class DataModel : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(DataModel)

  Property(Graph, workflow::Graph*)

private:
  static DataModel* instance;

protected:
  DataModel(void);

public:
  virtual ~DataModel(void);

  static DataModel& getInstance();
};

}

}

#endif
