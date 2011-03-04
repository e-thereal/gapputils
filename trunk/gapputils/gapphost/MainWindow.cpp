#include "MainWindow.h"

#include <qaction.h>
#include <qmenubar.h>
#include <qtablewidget.h>
#include <qsplitter.h>
#include <qlabel.h>

namespace gapputils {

namespace host {

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
  setWindowTitle("gapputils Application Host");
  fileMenu = menuBar()->addMenu("File");
  QAction* quitAction = fileMenu->addAction("Quit");

  connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  QLabel* helloLabel = new QLabel("Hello", this);
  QTableWidget* table = new QTableWidget(3, 2);

  QSplitter* splitter = new QSplitter(Qt::Orientations::enum_type::Horizontal);
  splitter->addWidget(helloLabel);
  splitter->addWidget(table);
  setCentralWidget(splitter);

  centralWidget = splitter;
}

MainWindow::~MainWindow()
{
  delete centralWidget;
  delete fileMenu;
}

void MainWindow::quit() {
  this->close();
}

}

}
