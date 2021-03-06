#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QMainWindow>

#include <qmenu.h>
#include <qtabwidget.h>
#include <qtimer.h>
#include <qsplitter.h>
#include "Workflow.h"

#include <map>
#include <qmdiarea.h>


namespace capputils {

namespace reflection {

class ReflectableClass;

}

}

namespace gapputils {

namespace host {

class WorkflowToolBox;
class PropertyGrid;
class WorkbenchWindow;
class WorkflowSnippets;
class GlobalPropertiesView;
class ModuleHelpWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QMenu* editMenu;
  QMenu* runMenu;
  QMenu* windowMenu;
  QAction* abortAction;
  WorkflowToolBox* toolBox;
  WorkflowSnippets* snippets;
  PropertyGrid* propertyGrid;
  ModuleHelpWidget* moduleHelp;
  GlobalPropertiesView* globalPropertiesView;
  QMdiArea* area;

  QTimer autoSaveTimer;
//  bool libsChanged;
  bool autoQuit;  ///< quits the program when the workflow has been updated.
  WorkbenchWindow* workingWindow;

  boost::shared_ptr<workflow::Workflow> grandpa;

public:
  // changed Qt::WFlags to Qt::WindowFlags
  MainWindow(QWidget *parent = 0, Qt::WindowFlags flags = 0);
  virtual ~MainWindow();

  virtual void closeEvent(QCloseEvent *event);
  void setGuiEnabled(bool enabled);
  void resume();
  void setAutoQuit(bool autoQuit);
  WorkbenchWindow* showWorkflow(boost::shared_ptr<workflow::Workflow> workflow);
  void saveWorkflowList();
  WorkbenchWindow* getCurrentWorkbenchWindow();

public Q_SLOTS:
  void quit();
  void loadWorkflow();
  void saveWorkflow();
  void save();
  void saveAs();
  void loadLibrary();
  void reload();

  void autoSave();
  void copy();
  void copyDanglingEdges();
  void paste();
  void removeSelectedItems();
  void createSnippet();
  void resetInputs();
  void incrementInputs();
  void decrementInputs();

  void updateCurrentModule();
  void updateWorkflow();
  void updateMainWorkflow();
  void updateMainWorkflowNode(const std::string& nodeLabel);
  void updateMainWorkflowNodes(const std::vector<std::string>& nodeLabels);
  void updateCurrentWorkflowNode(const capputils::reflection::ReflectableClass* object);
  void terminateUpdate();
  void updateFinished();
  
  //void showWorkflow(boost::shared_ptr<workflow::Workflow> workflow, bool addUuid = true);
  void closeWorkflow(const std::string& uuid);
  void subWindowActivated(QMdiSubWindow* window);
  void handleCurrentNodeChanged(boost::shared_ptr<workflow::Node> node);
  void selectModule(const QString& uuid);
};

}

}

#endif // GAPPHOST_H
