#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <ReflectableClass.h>

namespace gapputils {

class ToolItem;

namespace workflow {

class Node : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(Node)

  Property(Uuid, std::string)
  Property(X, int)
  Property(Y, int)
  Property(Module, capputils::reflection::ReflectableClass*)
  Property(ToolItem, ToolItem*)

public:
  Node();
  virtual ~Node(void);
};

}

}

#endif
