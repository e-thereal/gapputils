#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <ReflectableClass.h>
#include <ObservableClass.h>

namespace gapputils {

class ToolItem;

namespace workflow {

class Node : public capputils::reflection::ReflectableClass,
             public capputils::ObservableClass
{

  InitReflectableClass(Node)

  Property(Uuid, std::string)
  Property(X, int)
  Property(Y, int)
  Property(Module, capputils::reflection::ReflectableClass*)
  Property(ToolItem, ToolItem*)
  Property(UpToDate, bool)

public:
  Node();
  virtual ~Node(void);

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
