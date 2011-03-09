#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include <qstandarditemmodel.h>
#include <qlabel.h>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QWidget* centralWidget;
  QLabel* testLabel;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

private slots:
  void quit();
  void itemChanged(QStandardItem* item);
};

}

}

#endif // GAPPHOST_H
