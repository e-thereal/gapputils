#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include "NewObjectDialog.h"
#include <qtabwidget.h>
#include <qtimer.h>
#include <qtreewidget.h>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  NewObjectDialog* newObjectDialog;
  QTabWidget* tabWidget;
  QTreeWidget* toolBox;
  QTimer reloadTimer;
  bool libsChanged;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

private Q_SLOTS:
  void quit();
  void newItem();
  void loadWorkflow();
  void saveWorkflow();
  void loadLibrary();
  void reload();
  void checkLibraryUpdates();
  void itemDoubleClickedHandler(QTreeWidgetItem *item, int column);
  void itemClickedHandler(QTreeWidgetItem *item, int column);
};

}

}

#endif // GAPPHOST_H
