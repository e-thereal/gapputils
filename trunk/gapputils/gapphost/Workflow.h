/*
 * Workflow.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_WORKFLOW_H_
#define GAPPHOST_WORKFLOW_H_

#include <capputils/ObservableClass.h>

#include <vector>
//#include <tinyxml/tinyxml.h>
#include <set>
#include "Edge.h"
#include "Node.h"
#include "GlobalProperty.h"
#include "GlobalEdge.h"
#include "PropertyReference.h"
#include <stack>
#include <capputils/TimedClass.h>
//#include "linreg.h"
//#include <ctime>

#include "Workbench.h"

//#include <boost/enable_shared_from_this.hpp>

class QStandardItem;

namespace capputils {
  class Logbook;
}

namespace gapputils {

class ToolItem;
class CableItem;
class ToolConnection;

namespace host {
class WorkflowUpdater;
}

namespace workflow {

class Workflow : public QObject, public Node, public CompatibilityChecker, public capputils::TimedClass
{
  Q_OBJECT

  InitReflectableClass(Workflow)

  Property(Libraries, boost::shared_ptr<std::vector<std::string> >)
  Property(Edges, boost::shared_ptr<std::vector<boost::shared_ptr<Edge> > >)
  Property(Nodes, boost::shared_ptr<std::vector<boost::shared_ptr<Node> > >)
  Property(GlobalProperties, boost::shared_ptr<std::vector<boost::shared_ptr<GlobalProperty> > >)
  Property(GlobalEdges, boost::shared_ptr<std::vector<boost::shared_ptr<GlobalEdge> > >)

  Property(ViewportScale, double)
  Property(ViewportPosition, std::vector<double>)
  Property(Logbook, boost::shared_ptr<capputils::Logbook>)

private:
  std::set<std::string> loadedLibraries;
  static int librariesId;
  std::vector<boost::shared_ptr<Node> > interfaceNodes;

public:
  Workflow();
  virtual ~Workflow();

  boost::shared_ptr<Workflow> shared_from_this()       { return boost::static_pointer_cast<Workflow>(Node::shared_from_this()); }
  boost::shared_ptr<const Workflow> shared_from_this() const { return boost::static_pointer_cast<const Workflow>(Node::shared_from_this()); }

  virtual void resume();
  void resumeNode(boost::shared_ptr<Node> node);
  bool resumeEdge(boost::shared_ptr<Edge> edge);

  bool isInputNode(boost::shared_ptr<const Node> node) const;
  bool isOutputNode(boost::shared_ptr<const Node> node) const;
  void getDependentNodes(boost::shared_ptr<Node> node, std::vector<boost::shared_ptr<Node> >& dependendNodes);
  bool isDependentProperty(boost::shared_ptr<const Node> node, const std::string& propertyName) const;

  void addInterfaceNode(boost::shared_ptr<Node> node);
  void removeInterfaceNode(boost::shared_ptr<Node> node);
  bool hasCollectionElementInterface() const;

  // id is as set by ToolItem (propertyCount + pos)
  boost::shared_ptr<const Node> getInterfaceNode(int id) const;
  std::vector<boost::shared_ptr<Node> >& getInterfaceNodes();

  /// Returns true if the propertyName was found
  bool getToolConnectionId(boost::shared_ptr<const Node> node, const std::string& propertyName, unsigned& id) const;

  boost::shared_ptr<Node> getNode(ToolItem* item);
  boost::shared_ptr<Node> getNode(ToolItem* item, unsigned& pos);
  boost::shared_ptr<Node> getNode(boost::shared_ptr<capputils::reflection::ReflectableClass> object);
  boost::shared_ptr<Node> getNode(boost::shared_ptr<capputils::reflection::ReflectableClass> object, unsigned& pos);
  boost::shared_ptr<Node> getNode(const std::string& uuid) const;

  boost::shared_ptr<Edge> getEdge(CableItem* cable);
  boost::shared_ptr<Edge> getEdge(CableItem* cable, unsigned& pos);

  boost::shared_ptr<const Node> getNode(ToolItem* item) const;
  boost::shared_ptr<const Node> getNode(ToolItem* item, unsigned& pos) const;

  boost::shared_ptr<const Edge> getEdge(CableItem* cable) const;
  boost::shared_ptr<const Edge> getEdge(CableItem* cable, unsigned& pos) const;

  boost::shared_ptr<GlobalProperty> getGlobalProperty(const std::string& name);
  boost::shared_ptr<GlobalProperty> getGlobalProperty(const PropertyReference& reference);
  boost::shared_ptr<GlobalEdge> getGlobalEdge(const PropertyReference& reference);

  virtual bool areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const;

  void makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference);
  void connectProperty(const std::string& name, const PropertyReference& propertyReference);
  void removeGlobalEdge(boost::shared_ptr<GlobalEdge> edge);
  void removeGlobalProperty(boost::shared_ptr<GlobalProperty> gprop);

  bool activateGlobalEdge(boost::shared_ptr<GlobalEdge> edge);

  void resetInputs();
  void incrementInputs();
  void decrementInputs();

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);
  void interfaceChangedHandler(capputils::ObservableClass* sender, int eventId);

public Q_SLOTS:
  void removeNode(boost::shared_ptr<Node> node);
  void removeEdge(boost::shared_ptr<Edge> edge);
};

}

}

#endif /* GAPPHOST_WORKFLOW_H_ */
