#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include "NewObjectDialog.h"
#include <qtabwidget.h>
#include <qtimer.h>
#include <qtreewidget.h>
#include <qsplitter.h>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QMenu* runMenu;
  NewObjectDialog* newObjectDialog;
  QTabWidget* tabWidget;
  QTreeWidget* toolBox;
  QTimer reloadTimer;
  bool libsChanged;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

  virtual void closeEvent(QCloseEvent *event);

private Q_SLOTS:
  void quit();
  void newItem();
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
  void updateFinished();
};

}

}

#endif // GAPPHOST_H
