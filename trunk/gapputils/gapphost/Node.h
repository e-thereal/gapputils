#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <gapputils/IProgressMonitor.h>
#include "ModelHarmonizer.h"

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

private:
  ModelHarmonizer* harmonizer;
  static int moduleId;

public:
  Node();
  virtual ~Node(void);

  virtual bool isUpToDate() const;
  virtual void update(IProgressMonitor* monitor);
  virtual void writeResults();

  QStandardItemModel* getModel();

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
