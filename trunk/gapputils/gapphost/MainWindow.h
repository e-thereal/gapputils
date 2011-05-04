#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include "NewObjectDialog.h"
#include <qtabwidget.h>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  NewObjectDialog* newObjectDialog;
  QTabWidget* tabWidget;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

private Q_SLOTS:
  void quit();
  void newItem();
  void loadWorkflow();
};

}

}

#endif // GAPPHOST_H
