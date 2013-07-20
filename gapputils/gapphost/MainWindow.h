#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include <qtabwidget.h>
#include <qtimer.h>
#include <qsplitter.h>
#include "Workflow.h"

#include <map>
#include <qmdiarea.h>

namespace gapputils {

namespace host {

class WorkflowToolBox;
class PropertyGrid;
class WorkbenchWindow;
class WorkflowSnippets;
class GlobalPropertiesView;

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
  GlobalPropertiesView* globalPropertiesView;
  QMdiArea* area;

  QTimer autoSaveTimer;
//  bool libsChanged;
  bool autoQuit;  ///< quits the program when the workflow has been updated.
  WorkbenchWindow* workingWindow;

  boost::shared_ptr<workflow::Workflow> grandpa;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
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
  void paste();
  void createSnippet();
  void resetInputs();
  void incrementInputs();
  void decrementInputs();

  void updateCurrentModule();
  void updateWorkflow();
  void updateMainWorkflow();
  void updateMainWorkflowNode(const std::string& nodeLabel);
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
