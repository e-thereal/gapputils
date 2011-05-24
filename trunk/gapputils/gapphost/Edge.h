#pragma once

#ifndef _GAPPHOST_EDGE_H_
#define _GAPPHOST_EDGE_H_

#include <capputils/ReflectableClass.h>

#include "Node.h"

namespace gapputils {

class CableItem;

namespace workflow {

class Edge : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(Edge)

  Property(OutputNode, std::string)
  Property(OutputProperty, std::string)
  Property(InputNode, std::string)
  Property(InputProperty, std::string)
  Property(CableItem, CableItem*)

public:
  Edge(void);
  virtual ~Edge(void);
};

}

}

#endif
