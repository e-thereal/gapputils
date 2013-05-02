/*
 * GlobalEdge.h
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_GLOBALEDGE_H_
#define GAPPHOST_GLOBALEDGE_H_

#include "Edge.h"

namespace gapputils {

namespace workflow {

class GlobalEdge : public Edge {
  InitReflectableClass(GlobalEdge)

  Property(GlobalProperty, std::string)

private:
  capputils::EventHandler<GlobalEdge> handler;
  int inputId; // TODO: get rid of inputId

public:
  GlobalEdge();
  virtual ~GlobalEdge();

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
};

}

}

#endif /* GAPPHOST_GLOBALEDGE_H_ */
