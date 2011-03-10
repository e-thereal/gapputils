#include "MainWindow.h"

#include <qaction.h>
#include <qmenubar.h>
#include <qtablewidget.h>
#include <qsplitter.h>
#include <qlabel.h>
#include <qtreeview.h>
#include <qstandarditemmodel.h>
#include <QWidgetItem>
#include <qcombobox.h>
#include <qtreewidget.h>
#include <qdatawidgetmapper.h>
#include <qitemdelegate.h>
#include <qvariant.h>

#include <DescriptionAttribute.h>
#include <ReflectableAttribute.h>
#include <ClassProperty.h>
#include <ScalarAttribute.h>

#include "Person.h"
#include "PropertyReference.h"
#include "PropertyGridDelegate.h"

using namespace std;
using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;

namespace gapputils {

namespace host {

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
  setWindowTitle("Application Host");
  fileMenu = menuBar()->addMenu("File");
  QAction* quitAction = fileMenu->addAction("Quit");

  connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  testLabel = new QLabel("Hello", this);
  testLabel->setGeometry(0, 0, 640, 600);
  //testLabel->setSizePolicy(Qt::Size);
  this->setGeometry(150, 150, 800, 600);

  harmonizer1 = new ModelHarmonizer(&person);
  harmonizer2 = new ModelHarmonizer(&person);
  
  QTreeView* tree = new QTreeView();
  tree->setAllColumnsShowFocus(false);
  tree->setAlternatingRowColors(true);
  tree->setSelectionBehavior(QAbstractItemView::SelectItems);
  tree->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  tree->setItemDelegate(new PropertyGridDelegate());
  tree->setModel(harmonizer1->getModel());
  tree->setItemDelegate(new PropertyGridDelegate());

  QTreeView* tree2 = new QTreeView();
  tree2->setAllColumnsShowFocus(false);
  tree2->setAlternatingRowColors(true);
  tree2->setSelectionBehavior(QAbstractItemView::SelectItems);
  tree2->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  tree2->setModel(harmonizer2->getModel());
  tree->setItemDelegate(new PropertyGridDelegate());

  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(testLabel);
  splitter->addWidget(tree);
  splitter->addWidget(tree2);
  setCentralWidget(splitter);

  centralWidget = splitter;
}

MainWindow::~MainWindow()
{
  delete centralWidget;
  delete fileMenu;
  delete harmonizer1;
  delete harmonizer2;
}

void MainWindow::quit() {
  this->close();
}

}

}
