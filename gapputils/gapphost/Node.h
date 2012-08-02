#pragma once

#ifndef _GAPPHOST_NODE_H_
#define _GAPPHOST_NODE_H_

#include <gapputils/gapputils.h>

#include <capputils/ReflectableClass.h>
#include <capputils/ObservableClass.h>
#include <gapputils/IProgressMonitor.h>

#include <boost/enable_shared_from_this.hpp>

#include "Expression.h"

class PropertyReference;

namespace gapputils {

class ToolItem;

namespace workflow {

class Workflow;
class GlobalProperty;
class GlobalEdge;

class Node : public capputils::reflection::ReflectableClass,
             public capputils::ObservableClass,
             public boost::enable_shared_from_this<Node>
{
public:
  InitReflectableClass(Node)

  Property(Uuid, std::string)
  Property(X, int)
  Property(Y, int)
  Property(Module, boost::shared_ptr<capputils::reflection::ReflectableClass>)
  Property(InputChecksum, checksum_t)
  Property(OutputChecksum, checksum_t)
  Property(ToolItem, ToolItem*)
  Property(Workflow, boost::weak_ptr<Workflow>)
  Property(Expressions, boost::shared_ptr<std::vector<boost::shared_ptr<Expression> > >)

private:
  static int moduleId;
  bool readFromCache;

public:
  Node();
  virtual ~Node(void);

  static std::string CreateUuid();

  void getDependentNodes(std::vector<boost::shared_ptr<Node> >& dependendNodes);

  /**
   * \brief Returns the expression object of the named property if the property is associated to one.
   *
   * \param[in] propertyName  Name of the property for which an expression is looked for
   *
   * \return The expression object, or 0 if the property is not bound to an expression
   */
  boost::shared_ptr<Expression> getExpression(const std::string& propertyName);

  /**
   * \brief Removes an expression from a property
   *
   * \param[in] propertyName  Name of the property for which the expression should be removed
   *
   * \return  True if an expression was removed. False if no expression for the given property could be found.
   */
  bool removeExpression(const std::string& propertyName);

  virtual void resume();
  void resumeExpressions();
  bool isDependentProperty(const std::string& propertyName) const;

//  QStandardItemModel* getModel();

  virtual boost::shared_ptr<GlobalProperty> getGlobalProperty(const PropertyReference& reference);
  virtual boost::shared_ptr<GlobalEdge> getGlobalEdge(const PropertyReference& reference);

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif
