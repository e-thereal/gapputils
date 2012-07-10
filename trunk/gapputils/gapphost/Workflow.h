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
#include <qformlayout.h>
#include <vector>
#include <qtreeview.h>
#include <tinyxml/tinyxml.h>
#include <set>
#include "Edge.h"
#include "Node.h"
#include "GlobalProperty.h"
#include "GlobalEdge.h"
#include "WorkflowWorker.h"
#include "PropertyReference.h"
#include <stack>
#include <capputils/TimedClass.h>
#include "linreg.h"
#include <ctime>

#include "Workbench.h"

#include <gapputils/InterfaceDescription.h>

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

  Property(Libraries, std::vector<std::string>*)
  Property(Edges, std::vector<Edge*>*)
  Property(Nodes, std::vector<Node*>*)
  Property(GlobalProperties, std::vector<GlobalProperty*>*)
  Property(GlobalEdges, std::vector<GlobalEdge*>*)

  Property(ViewportScale, double)
  Property(ViewportPosition, std::vector<double>)

private:
  Workbench* workbench;
  QTreeView* propertyGrid;
  QFormLayout* infoLayout;
  QWidget* widget;
  std::set<std::string> loadedLibraries;
  bool ownWidget,                         ///< True at the beginning. False if widget has been dispensed
       processingCombination,             ///< True if in combination mode of a CombinerInterface
       dryrun;                            ///< True if in dry-run mode. This mode calculates checksums only
                                          ///  in order the check if an update of the output parameters is
                                          ///  necessary. Intermediate nodes are not updated when the outputs
                                          ///  don't require an update

  WorkflowWorker* worker;
  std::stack<Node*> nodeStack;
  std::stack<Node*> processedStack;
  std::set<Node*> processedNodes;
  static int librariesId;
  QAction *makeGlobal, *removeGlobal, *connectToGlobal, *disconnectFromGlobal;
  Node* progressNode;
  LinearRegression etaRegression;
  time_t startTime;
  std::vector<Node*> interfaceNodes;
  boost::shared_ptr<host::WorkflowUpdater> workflowUpdater;

public:
  Workflow();
  virtual ~Workflow();

  void newItem(Node* node);
  bool newCable(Edge* edge);
  virtual void resume();
  void resumeViewport();
  void resumeNode(Node* node);

  bool isInputNode(const Node* node) const;
  bool isOutputNode(const Node* node) const;
  void getDependentNodes(Node* node, std::vector<Node*>& dependendNodes);
  bool isDependentProperty(const Node* node, const std::string& propertyName) const;

  /// The workflow loses ownership of the widget when calling this method
  QWidget* dispenseWidget();

  /// This call is asynchronous. updateFinished signal is emitted when done.
  void updateCurrentModule();

  /**
   * \brief Updates all outputs
   *
   * \param[in] updateNodes If true, all nodes that require an update are updated,
   *                        otherwise an update is only performed if the output parameters
   *                        need an update.
   */
  void updateOutputs(bool updateNodes = false);
  void abortUpdate();
  void processStack();
  void buildStack(Node* node);
  //void load(const std::string& filename);

  void copySelectedNodesToClipboard();
  void addNodesFromClipboard();

  void addInterfaceNode(Node* node);
  void removeInterfaceNode(Node* node);
  bool hasCollectionElementInterface() const;

  // id is as set by ToolItem (propertyCount + pos + 1)
  const Node* getInterfaceNode(int id) const;
  std::vector<Node*>& getInterfaceNodes();

  /// Returns the name of the property if connectionId refers to a property of the
  /// workflows module. Otherwise, it is assumed that connectionId refers to an
  /// interface node. In that case, the Uuid of the interface node is return.
  /// If the connectionId is not valid, an empty string is returned.
  std::string getPropertyName(const Node* node, int connectionId) const;
  virtual PropertyReference* getPropertyReference(const std::string& propertyName);
  virtual ConstPropertyReference* getPropertyReference(const std::string& propertyName) const;

  /// Returns the true if the propertyName was found
  bool getToolConnectionId(const Node* node, const std::string& propertyName, unsigned& id) const;

  virtual void updateCache();
  virtual bool restoreFromCache();

  /**
   * \brief Updates all nodes that need an update.
   *
   * This method assumes that updateChecksum was called before.
   */
  void updateNodes();

  /// Workflows are never up-to-date unless all modules are up-to-date
  /**
   * Checking if a workflow is up-to-date involves individual checks of each module
   * Checking each modules is the same as if you would recalculate the workflow. Thus,
   * workflows are always assumed not to be up to date.
   */
  virtual void update(IProgressMonitor* monitor, bool force);
  virtual void writeResults();

  void setUiEnabled(bool enabled);

  Node* getNode(ToolItem* item);
  Node* getNode(ToolItem* item, unsigned& pos);
  Node* getNode(capputils::reflection::ReflectableClass* object);
  Node* getNode(capputils::reflection::ReflectableClass* object, unsigned& pos);
  Node* getNode(const std::string& uuid);

  Edge* getEdge(CableItem* cable);
  Edge* getEdge(CableItem* cable, unsigned& pos);

  const Node* getNode(ToolItem* item) const;
  const Node* getNode(ToolItem* item, unsigned& pos) const;

  const Edge* getEdge(CableItem* cable) const;
  const Edge* getEdge(CableItem* cable, unsigned& pos) const;

  GlobalProperty* getGlobalProperty(const std::string& name);
  GlobalProperty* getGlobalProperty(capputils::reflection::ReflectableClass* object,
    capputils::reflection::IClassProperty* property);
  GlobalEdge* getGlobalEdge(capputils::reflection::ReflectableClass* object,
      capputils::reflection::IClassProperty* property);

  QStandardItem* getItem(capputils::reflection::ReflectableClass*,
      capputils::reflection::IClassProperty* property);

  // Returns null if current item is not a workflow
  Workflow* getCurrentWorkflow();

  virtual bool areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const;

  void makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference);
  void connectProperty(const std::string& name, const PropertyReference& propertyReference);
  void removeGlobalEdge(GlobalEdge* edge);
  void removeGlobalProperty(GlobalProperty* gprop);

  // This method activates a global property. I.e. it fills the runtime values of the
  // property object and updates the graphical appearance. Returns false if property could
  // not be activated
  bool activateGlobalProperty(GlobalProperty* prop);

  // This method deactivates a global property. I.e. it updates the graphical
  // appearance in the property grid to reflect that a property is no longer global.
  void deactivateGlobalProperty(GlobalProperty* prop);

  void activateGlobalEdge(GlobalEdge* edge);

private:
  void changedHandler(capputils::ObservableClass* sender, int eventId);

private Q_SLOTS:
  void createModule(int x, int y, QString classname);
  void deleteModule(ToolItem* item);
  void itemSelected(ToolItem* item);

  void removeEdge(Edge* edge);

  void itemChangedHandler(ToolItem* item);
  void createEdge(CableItem* cable);
  void deleteEdge(CableItem* cable);

  void finalizeModuleUpdate(workflow::Node* node);
  void showProgress(workflow::Node* node, double progress, bool updateNode);
  void showWorkflow(workflow::Workflow* workflow);
  void showWorkflow(ToolItem* item);
  void showModuleDialog(ToolItem* item);
  void delegateDeleteCalled(workflow::Workflow* workflow);
  void handleViewportChanged();

  void showContextMenu(const QPoint &);
  void gridClicked(const QModelIndex& index);
  void makePropertyGlobal();
  void removePropertyFromGlobal();
  void connectProperty();
  void disconnectProperty();

  void workflowUpdateFinished();

Q_SIGNALS:
  void updateFinished(workflow::Node* node);
  void processModule(workflow::Node* node, bool force);
  void showWorkflowRequest(workflow::Workflow* workflow);
  void deleteCalled(workflow::Workflow* workflow);
};

}

}

#endif /* GAPPHOST_WORKFLOW_H_ */
