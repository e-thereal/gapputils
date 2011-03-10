#ifndef GAPPHOST_H
#define GAPPHOST_H

#include <QtGui/QMainWindow>

#include <qmenu.h>
#include <qstandarditemmodel.h>
#include <qlabel.h>
#include "Person.h"
#include "ModelHarmonizer.h"

namespace gapputils {

namespace host {

class MainWindow : public QMainWindow
{
    Q_OBJECT

private:
  QMenu* fileMenu;
  QWidget* centralWidget;
  QLabel* testLabel;
  Person person;
  ModelHarmonizer* harmonizer1;
  ModelHarmonizer* harmonizer2;

public:
  MainWindow(QWidget *parent = 0, Qt::WFlags flags = 0);
  virtual ~MainWindow();

private Q_SLOTS:
  void quit();
};

}

}

#endif // GAPPHOST_H
