/*
 * WorkbenchWindow.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_WORKBENCHWINDOW_H_
#define GAPPUTILS_HOST_WORKBENCHWINDOW_H_

#include <qmdisubwindow.h>

#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <capputils/EventHandler.h>
#include <capputils/reflection/ReflectableClass.h>

#include <ctime>
#include <set>

#include "linreg.h"

namespace gapputils {

class Workbench;
class ToolItem;
class CableItem;

namespace workflow {
  class Workflow;
  class Node;
  class Edge;
}

namespace host {

  class WorkflowUpdater;

class WorkbenchWindow : public QMdiSubWindow {

  Q_OBJECT

private:
  boost::weak_ptr<workflow::Workflow> workflow;
  Workbench* workbench;
  boost::shared_ptr<WorkflowUpdater> workflowUpdater;
  std::set<boost::weak_ptr<workflow::Node> > processedNodes;
  boost::weak_ptr<workflow::Node> progressNode;
  LinearRegression etaRegression;
  time_t startTime;
  capputils::EventHandler<WorkbenchWindow> handler, modelEventHandler;
  bool closable;

public:
  WorkbenchWindow(boost::shared_ptr<workflow::Workflow> workflow, QWidget* parent = 0);
  virtual ~WorkbenchWindow();

  boost::shared_ptr<workflow::Workflow> getWorkflow() const;

  void setClosable(bool closable);
  bool getClosable() const;

  void createItem(boost::shared_ptr<workflow::Node> node);
  bool createCable(boost::shared_ptr<workflow::Edge> edge);
  void removeSelectedItems();

  boost::shared_ptr<workflow::Workflow> copySelectedNodes(bool copyDanglingEdges = false);
  void addNodes(workflow::Workflow& pasteWorkflow);
  void copySelectedNodesToClipboard(bool copyDanglingEdges = false);
  void addNodesFromClipboard();
  void createSnippet();
  void addNodesFromSnippet(int x, int y, const std::string& filename);

  void setUiEnabled(bool enabled);
  void resumeViewport();
  bool trySelectNode(const std::string& uuid);
  boost::shared_ptr<workflow::Node> getCurrentNode();

  // Update stuff
  void updateCurrentModule();
  void updateInputs();
  void updateOutputs();
  void updateNodeByLabel(const std::string& label);
  void updateNodesByLabels(const std::vector<std::string>& labels);
  void updateNode(const capputils::reflection::ReflectableClass* object);
  void abortUpdate();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
  void handleModelEvents(capputils::ObservableClass* sender, int eventId);

protected:
  void closeEvent(QCloseEvent *event);

public Q_SLOTS:
  boost::shared_ptr<workflow::Node> createModule(int x, int y, QString classname);
  void createEdge(CableItem* cable, int position);
  void deleteEdge(CableItem* cable);
  void deleteModule(ToolItem* item);

  void itemChangedHandler(ToolItem* item);
  void itemSelected(ToolItem* item);
  void showModuleDialog(ToolItem* item);
  void showWorkflow(ToolItem* item);
  void handleViewportChanged();

  void showProgress(boost::shared_ptr<workflow::Node> node, double progress);
  void workflowUpdateFinished();

Q_SIGNALS:
  void updateFinished();
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_WORKBENCHWINDOW_H_ */
