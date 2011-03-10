#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include <qstandarditemmodel.h>
#include "Person.h"
#include "ModelHarmonizer.h"
#include <qtreeview.h>

namespace gapputils {

class Workbench;
class ToolItem;

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  Workbench* bench;
  QTreeView* propertyGrid;
  QWidget* centralWidget;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

private Q_SLOTS:
  void quit();
  void newItem();
  void itemSelected(ToolItem* item);
};

}

}

#endif // GAPPHOST_H
