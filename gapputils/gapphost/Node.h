#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <gapputils/IProgressMonitor.h>

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

public:
  Node();
  virtual ~Node(void);

  virtual bool isUpToDate() const;
  virtual void update(IProgressMonitor* monitor);
  virtual void writeResults();

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
