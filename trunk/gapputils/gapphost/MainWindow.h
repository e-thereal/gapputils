#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include "NewObjectDialog.h"
#include <qtabwidget.h>
#include <qtimer.h>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  NewObjectDialog* newObjectDialog;
  QTabWidget* tabWidget;
  QTimer reloadTimer;

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
};

}

}

#endif // GAPPHOST_H
