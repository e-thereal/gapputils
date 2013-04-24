/*
 * Expression.h
 *
 *  Created on: Jan 27, 2012
 *      Author: tombr
 */

#ifndef GAPPUTLIS_WORKFLOW_EXPRESSION_H_
#define GAPPUTLIS_WORKFLOW_EXPRESSION_H_

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <capputils/EventHandler.h>

#include <set>

#include <boost/enable_shared_from_this.hpp>

namespace gapputils {

namespace workflow {

class Node;
class GlobalProperty;

class Expression : public capputils::reflection::ReflectableClass,
                   public boost::enable_shared_from_this<Expression>
{

  InitReflectableClass(Expression)

  Property(Expression, std::string)
  Property(PropertyName, std::string)
  Property(Node, boost::weak_ptr<Node>)

private:
  capputils::EventHandler<Expression> handler;
  std::set<std::pair<capputils::ObservableClass*, int> > observedProperties;
  std::set<boost::weak_ptr<GlobalProperty> > globalProperties;

public:
  Expression();
  virtual ~Expression();

  std::string evaluate() const;

  bool resume();
  void disconnect(boost::shared_ptr<GlobalProperty> gprop);
  void disconnectAll();
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

} /* namespace workflow */

} /* namespace gapputils */

#endif /* GAPPUTLIS_WORKFLOW_EXPRESSION_H_ */
