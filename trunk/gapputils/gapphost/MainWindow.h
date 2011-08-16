#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include "NewObjectDialog.h"
#include <qtabwidget.h>
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
  NewObjectDialog* newObjectDialog;
  QTabWidget* tabWidget;
  QTreeWidget* toolBox;
  QTimer reloadTimer;
  bool libsChanged;
  std::vector<workflow::Workflow*> openWorkflows;
  QAction* changeInterfaceAction;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

  virtual void closeEvent(QCloseEvent *event);
  void setGuiEnabled(bool enabled);

private Q_SLOTS:
  void quit();
  void loadWorkflow();
  void saveWorkflow();
  void save();
  void loadLibrary();
  void reload();
  void checkLibraryUpdates();
  void itemDoubleClickedHandler(QTreeWidgetItem *item, int column);
  void itemClickedHandler(QTreeWidgetItem *item, int column);

  void updateCurrentModule();
  void updateWorkflow();
  void terminateUpdate();
  void editCurrentInterface();
  void updateEditMenuStatus();
  void updateFinished(workflow::Node* node);
  void showWorkflow(workflow::Workflow* workflow);
  void closeWorkflow(workflow::Workflow* workflow);
  void closeWorkflow(int tabIndex);
};

}

}

#endif // GAPPHOST_H
