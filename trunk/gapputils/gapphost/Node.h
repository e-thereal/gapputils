#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <ReflectableClass.h>

namespace gapputils {

namespace workflow {

class Node : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(Node)

  Property(X, int)
  Property(Y, int)
  Property(Module, capputils::reflection::ReflectableClass*)

public:
  Node();
  virtual ~Node(void);
};

}

}

#endif
