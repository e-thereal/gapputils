/*
 * Workflow.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef GAPPHOST_WORKFLOW_H_
#define GAPPHOST_WORKFLOW_H_

#include <capputils/ObservableClass.h>

#include <qobject.h>
#include <qwidget.h>
#include <qaction.h>
#include <vector>
#include <tinyxml/tinyxml.h>
#include <set>
#include "Edge.h"
#include "Node.h"
#include "GlobalProperty.h"
#include "GlobalEdge.h"
#include "PropertyReference.h"
#include <stack>
#include <capputils/TimedClass.h>
#include "linreg.h"
#include <ctime>

#include "Workbench.h"

#include <boost/enable_shared_from_this.hpp>

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
  Workbench* workbench;
  QWidget* widget;
  std::set<std::string> loadedLibraries;
  bool ownWidget;                         ///< True at the beginning. False if widget has been dispensed

  std::set<boost::weak_ptr<Node> > processedNodes;
  static int librariesId;
  boost::weak_ptr<Node> progressNode;
  LinearRegression etaRegression;
  time_t startTime;
  std::vector<boost::shared_ptr<Node> > interfaceNodes;
  boost::shared_ptr<host::WorkflowUpdater> workflowUpdater;

public:
  Workflow();
  virtual ~Workflow();

  boost::shared_ptr<Workflow> shared_from_this()       { return boost::static_pointer_cast<Workflow>(Node::shared_from_this()); }
  boost::shared_ptr<const Workflow> shared_from_this() const { return boost::static_pointer_cast<const Workflow>(Node::shared_from_this()); }

//  void newItem(boost::shared_ptr<Node> node);
  virtual void resume();
  void resumeViewport();
  void resumeNode(boost::shared_ptr<Node> node);
  bool resumeEdge(boost::shared_ptr<Edge> edge);

  bool isInputNode(boost::shared_ptr<const Node> node) const;
  bool isOutputNode(boost::shared_ptr<const Node> node) const;
  void getDependentNodes(boost::shared_ptr<Node> node, std::vector<boost::shared_ptr<Node> >& dependendNodes);
  bool isDependentProperty(boost::shared_ptr<const Node> node, const std::string& propertyName) const;

  /// The workflow loses ownership of the widget when calling this method
  QWidget* dispenseWidget();

  /// This call is asynchronous. updateFinished signal is emitted when done.
  void updateCurrentModule();
  void updateOutputs();
  void abortUpdate();

  bool trySelectNode(const std::string& uuid);

  void copySelectedNodesToClipboard();
//  void addNodesFromClipboard();

  void addInterfaceNode(boost::shared_ptr<Node> node);
  void removeInterfaceNode(boost::shared_ptr<Node> node);
  bool hasCollectionElementInterface() const;

  // id is as set by ToolItem (propertyCount + pos)
  boost::shared_ptr<const Node> getInterfaceNode(int id) const;
  std::vector<boost::shared_ptr<Node> >& getInterfaceNodes();

  /// Returns the name of the property if connectionId refers to a property of the
  /// workflows module. Otherwise, it is assumed that connectionId refers to an
  /// interface node. In that case, the Uuid of the interface node is return.
  /// If the connectionId is not valid, an empty string is returned.
  std::string getPropertyName(boost::shared_ptr<const Node> node, int connectionId) const;

  /// Returns true if the propertyName was found
  bool getToolConnectionId(boost::shared_ptr<const Node> node, const std::string& propertyName, unsigned& id) const;

  void setUiEnabled(bool enabled);

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

  // Returns null if current item is not a workflow
  boost::shared_ptr<Node> getCurrentNode();

  virtual bool areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const;

  void makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference);
  void connectProperty(const std::string& name, const PropertyReference& propertyReference);
  void removeGlobalEdge(boost::shared_ptr<GlobalEdge> edge);
  void removeGlobalProperty(boost::shared_ptr<GlobalProperty> gprop);

//  // This method deactivates a global property. I.e. it updates the graphical
//  // appearance in the property grid to reflect that a property is no longer global.
//  void deactivateGlobalProperty(GlobalProperty* prop);

  void activateGlobalEdge(boost::shared_ptr<GlobalEdge> edge);

  void resetInputs();
  void incrementInputs();
  void decrementInputs();

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);

public Q_SLOTS:
//  void createModule(int x, int y, QString classname);
  void deleteModule(ToolItem* item);
  void itemSelected(ToolItem* item);

  void removeEdge(boost::shared_ptr<Edge> edge);

  void itemChangedHandler(ToolItem* item);
  void createEdge(CableItem* cable);
  void deleteEdge(CableItem* cable);

  void showProgress(boost::shared_ptr<workflow::Node> node, double progress);
  void showWorkflow(boost::shared_ptr<workflow::Workflow> workflow);
  void showWorkflow(ToolItem* item);
  void showModuleDialog(ToolItem* item);
  void delegateDeleteCalled(const std::string& uuid);
  void handleViewportChanged();

  void workflowUpdateFinished();

Q_SIGNALS:
  void updateFinished(boost::shared_ptr<workflow::Node> node);
  void showWorkflowRequest(boost::shared_ptr<workflow::Workflow> workflow);
  void deleteCalled(const std::string& uuid);
  void currentModuleChanged(boost::shared_ptr<workflow::Node> node);
};

}

}

#endif /* GAPPHOST_WORKFLOW_H_ */
