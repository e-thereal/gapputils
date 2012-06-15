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

namespace gapputils {

namespace workflow {

class Node;
class GlobalProperty;

class Expression : public capputils::reflection::ReflectableClass {

  InitReflectableClass(Expression)

  Property(Expression, std::string)
  Property(PropertyName, std::string)
  Property(Node, Node*)

private:
  capputils::EventHandler<Expression> handler;
  std::set<std::pair<capputils::ObservableClass*, int> > observedProperties;
  std::set<GlobalProperty*> globalProperties;

public:
  Expression();
  virtual ~Expression();

  std::string evaluate() const;

  void resume();
  void disconnect(GlobalProperty* gprop);
  void disconnectAll();
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

} /* namespace workflow */

} /* namespace gapputils */

#endif /* GAPPUTLIS_WORKFLOW_EXPRESSION_H_ */
