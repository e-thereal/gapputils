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
#include <vector>
#include <qtreeview.h>
#include <tinyxml/tinyxml.h>
#include <set>
#include "Edge.h"
#include "Node.h"
#include "WorkflowWorker.h"
#include <stack>

namespace gapputils {

class Workbench;
class ToolItem;
class CableItem;

namespace workflow {

class Workflow : public QObject, public Node
{
  Q_OBJECT

  InitReflectableClass(Workflow)

  Property(Libraries, std::vector<std::string>*)
  Property(Edges, std::vector<Edge*>*)
  Property(Nodes, std::vector<Node*>*)
  Property(InputsPosition, std::vector<int>)
  Property(OutputsPosition, std::vector<int>)

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

public:
  Workflow();
  virtual ~Workflow();

  void newItem(Node* node);
  void newCable(Edge* edge);
  void resumeFromModel();
  /// The workflow loses ownership of the widget when calling this method
  QWidget* dispenseWidget();
  //TiXmlElement* getXml(bool addEmptyModule = true) const;

  /// This call is asynchronous. updateFinished signal is emitted when done.
  void updateSelectedModule();
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
  void delegateDeleteCalled(workflow::Workflow* workflow);

Q_SIGNALS:
  void updateFinished(workflow::Node* node);
  void processModule(workflow::Node* node);
  void showWorkflowRequest(workflow::Workflow* workflow);
  void deleteCalled(workflow::Workflow* workflow);
};

}

}

#endif /* WORKFLOW_H_ */
