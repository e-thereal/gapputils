#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include "NewObjectDialog.h"
#include <qtabwidget.h>
#include <qlineedit.h>
#include <qtimer.h>
#include <qtreewidget.h>
#include <qsplitter.h>
#include "Workflow.h"

#include <map>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QMenu* editMenu;
  QMenu* runMenu;
  QAction* abortAction;
  NewObjectDialog* newObjectDialog;
  QTabWidget* tabWidget;
  QLineEdit* toolBoxFilterEdit;
  QTreeWidget* toolBox;
  QTimer reloadTimer;
  bool libsChanged, autoQuit;  ///< quits the problem when the workflow has been updated.
  std::vector<workflow::Workflow*> openWorkflows;
  workflow::Workflow* workingWorkflow;
  std::map<QTreeWidgetItem*, boost::shared_ptr<std::vector<QTreeWidgetItem* > > > toolBoxItems;

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
  void itemClickedHandler(QTreeWidgetItem *item, int column);
  void copy();
  void paste();
  void focusFilter();
  void filterToolBox(const QString& text);

  void updateCurrentModule();
  void updateWorkflow();
  void updateMainWorkflow();
  void terminateUpdate();
  void updateFinished(workflow::Node* node);
  void showWorkflow(workflow::Workflow* workflow, bool addUuid = true);
  void closeWorkflow(workflow::Workflow* workflow);
  void closeWorkflow(int tabIndex);
  void currentTabChanged(int index);
};

}

}

#endif // GAPPHOST_H
