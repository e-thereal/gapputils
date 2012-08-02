/*
 * GlobalProperty.h
 *
 *  Created on: Jun 23, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_GLOBALPROPERTY_H_
#define GAPPHOST_GLOBALPROPERTY_H_

#include <capputils/ReflectableClass.h>

#include "Edge.h"
#include "Node.h"

namespace gapputils {

namespace workflow {

class Expression;

class GlobalProperty : public capputils::reflection::ReflectableClass {

  InitReflectableClass(GlobalProperty)

  Property(Name, std::string)
  Property(ModuleUuid, std::string)
  Property(PropertyId, std::string)
  Property(Edges, boost::shared_ptr<std::vector<boost::weak_ptr<Edge> > >)
  Property(Expressions, boost::shared_ptr<std::vector<boost::weak_ptr<Expression> > >)

public:
  GlobalProperty();
  virtual ~GlobalProperty();

  void addEdge(boost::shared_ptr<Edge> edge);
  void removeEdge(boost::shared_ptr<Edge> edge);
};

}

}

#endif /* GAPPHOST_GLOBALPROPERTY_H_ */
