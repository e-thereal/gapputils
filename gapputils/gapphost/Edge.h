#pragma once

#ifndef _GAPPHOST_EDGE_H_
#define _GAPPHOST_EDGE_H_

#include <capputils/EventHandler.h>
#include <capputils/ObservableClass.h>
#include <capputils/reflection/ReflectableClass.h>

#include "Node.h"

#include <boost/weak_ptr.hpp>

class PropertyReference;

namespace gapputils {

class CableItem;

namespace workflow {

class Node;

class Edge : public capputils::reflection::ReflectableClass,
             public capputils::ObservableClass
{

  InitReflectableClass(Edge)

  Property(OutputNode, std::string)
  Property(OutputProperty, std::string)
  // TODO: get rid of node pointers. Use property reference instead
  Property(OutputNodePtr, boost::weak_ptr<Node>)
  Property(OutputReference, boost::shared_ptr<PropertyReference>)

  Property(InputNode, std::string)
  Property(InputProperty, std::string)
  Property(InputNodePtr, boost::weak_ptr<Node>)
  Property(InputReference, boost::shared_ptr<PropertyReference>)
  Property(InputPosition, int)

  Property(CableItem, CableItem*)

private:
  capputils::EventHandler<Edge> handler;
  int outputId; // TODO: get rid of outputId
  static int positionId;

public:
  Edge(void);
  virtual ~Edge(void);

  /**
   * \brief Activates the edge if the properties are compatible
   *
   * \return Returns \c false iff properties are not compatible.
   *
   * Activating an edge means that from now on the values of the connected
   * properties are kept in sync. Compatible means that both properties are
   * of the same type or a type suitable type conversion is available.
   */
  // TODO: activate via workflow pointer: well create property references
  bool activate(boost::shared_ptr<Node> outputNode, boost::shared_ptr<Node> inputNode);
  void changedHandler(capputils::ObservableClass* sender, int eventId);

  static bool areCompatible(const capputils::reflection::IClassProperty* outputProperty,
      const capputils::reflection::IClassProperty* inputProperty);
};

}

}

#endif
