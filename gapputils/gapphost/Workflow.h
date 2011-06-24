/*
 * Workflow.h
 *
 *  Created on: May 3, 2011
 *      Author: tombr
 */

#ifndef WORKFLOW_H_
#define WORKFLOW_H_

#include <capputils/ObservableClass.h>

#include <qobject.h>
#include <qwidget.h>
#include <qaction.h>
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

#include "Workbench.h"

namespace gapputils {

class ToolItem;
class CableItem;
class ToolConnection;

namespace workflow {

class Workflow : public QObject, public Node, public CompatibilityChecker
{
  Q_OBJECT

  InitReflectableClass(Workflow)

  Property(Libraries, std::vector<std::string>*)
  Property(Edges, std::vector<Edge*>*)
  Property(Nodes, std::vector<Node*>*)
  Property(GlobalProperties, std::vector<GlobalProperty*>*)
  Property(GlobalEdges, std::vector<GlobalEdge*>*)

  Property(InputsPosition, std::vector<int>)
  Property(OutputsPosition, std::vector<int>)
  Property(ViewportScale, double)
  Property(ViewportPosition, std::vector<double>)

private:
  Workbench* workbench;
  QTreeView* propertyGrid;
  QWidget* widget;
  Node inputsNode, outputsNode;
  std::set<std::string> loadedLibraries;
  bool ownWidget, hasIONodes, processingCombination;
  WorkflowWorker* worker;
  std::stack<Node*> nodeStack;
  std::stack<Node*> processedStack;
  static int librariesId;
  QAction *makeGlobal, *removeGlobal, *connectToGlobal, *disconnectFromGlobal;

public:
  Workflow();
  virtual ~Workflow();

  void newItem(Node* node);
  void newCable(Edge* edge);
  void resumeFromModel();
  void resumeViewport();

  /// The workflow loses ownership of the widget when calling this method
  QWidget* dispenseWidget();
  //TiXmlElement* getXml(bool addEmptyModule = true) const;

  /// This call is asynchronous. updateFinished signal is emitted when done.
  void updateCurrentModule();
  void updateOutputs();
  void processStack();
  void buildStack(Node* node);
  void load(const std::string& filename);

  /// Workflows are never up-to-date unless all modules are up-to-date
  /**
   * Checking if a workflow is up-to-date involves individual checks of each module
   * Checking each modules is the same as if you would recalculate the workflow. Thus,
   * workflows are always assumed not to be up to date.
   */
  virtual bool isUpToDate() const;
  virtual void update(IProgressMonitor* monitor);
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

  QStandardItem* getItem(capputils::reflection::ReflectableClass*,
      capputils::reflection::IClassProperty* property);

  virtual bool areCompatibleConnections(const ToolConnection* output, const ToolConnection* input) const;

  void makePropertyGlobal(const std::string& name, const PropertyReference& propertyReference);
  void connectProperty(const std::string& name, const PropertyReference& propertyReference);

  // This method activates a global property. I.e. it fills the runtime values of the
  // property object and updates the graphical appearance
  void activateGlobalProperty(GlobalProperty* prop);

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
  void showProgress(workflow::Node* node, int i);
  void showWorkflow(workflow::Workflow* workflow);
  void showWorkflow(ToolItem* item);
  void delegateDeleteCalled(workflow::Workflow* workflow);
  void handleViewportChanged();

  void showContextMenu(const QPoint &);
  void makePropertyGlobal();
  void removePropertyFromGlobal();
  void connectProperty();
  void disconnectProperty();

Q_SIGNALS:
  void updateFinished(workflow::Node* node);
  void processModule(workflow::Node* node);
  void showWorkflowRequest(workflow::Workflow* workflow);
  void deleteCalled(workflow::Workflow* workflow);
};

}

}

#endif /* WORKFLOW_H_ */
