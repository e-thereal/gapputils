#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QWidget* centralWidget;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

private slots:
  void quit();
};

}

}

#endif // GAPPHOST_H
