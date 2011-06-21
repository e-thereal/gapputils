#pragma once

#ifndef _GAPPHOST_EDGE_H_
#define _GAPPHOST_EDGE_H_

#include <capputils/EventHandler.h>
#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>

#include "Node.h"

namespace gapputils {

class CableItem;

namespace workflow {

class Node;

class Edge : public capputils::reflection::ReflectableClass
{

  InitReflectableClass(Edge)

  Property(OutputNode, std::string)
  Property(OutputProperty, std::string)
  Property(OutputNodePtr, Node*)

  Property(InputNode, std::string)
  Property(InputProperty, std::string)
  Property(InputNodePtr, Node*)
  Property(CableItem, CableItem*)

private:
  capputils::EventHandler<Edge> handler;
  unsigned inputId, outputId;

public:
  Edge(void);
  virtual ~Edge(void);

  /**
   * \brief Activates the edge
   *
   * Activating an edge means that from now on the values of the connected
   * properties are kept in sync.
   */
  void activate(Node* outputNode, Node* inputNode);
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
