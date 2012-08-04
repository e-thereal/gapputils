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

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QMenu* editMenu;
  QMenu* runMenu;
  QMenu* windowMenu;
  QAction* abortAction;
  QTabWidget* tabWidget;
  WorkflowToolBox* toolBox;
  PropertyGrid* propertyGrid;
  QMdiArea* area;

  QTimer reloadTimer;
  bool libsChanged, autoQuit;  ///< quits the problem when the workflow has been updated.
  std::vector<boost::weak_ptr<workflow::Workflow> > openWorkflows;
  boost::weak_ptr<workflow::Workflow> workingWorkflow;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

  virtual void closeEvent(QCloseEvent *event);
  void setGuiEnabled(bool enabled);
  void resume();
  void setAutoQuit(bool autoQuit);

public Q_SLOTS:
  void quit();
  void loadWorkflow();
  void saveWorkflow();
  void save();
  void saveAs();
  void loadLibrary();
  void reload();
  void checkLibraryUpdates();
  void copy();
  void paste();
  void resetInputs();
  void incrementInputs();
  void decrementInputs();

  void updateCurrentModule();
  void updateWorkflow();
  void updateMainWorkflow();
  void terminateUpdate();
  void updateFinished(boost::shared_ptr<workflow::Node> node);
  void showWorkflow(boost::shared_ptr<workflow::Workflow> workflow, bool addUuid = true);
  void closeWorkflow(const std::string& uuid);
  void closeWorkflow(int tabIndex);
  void currentTabChanged(int index);
  void handleCurrentNodeChanged(boost::shared_ptr<workflow::Node> node);
  void selectModule(const QString& uuid);
};

}

}

#endif // GAPPHOST_H
