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
#include "Workbench.h"
#include "ToolItem.h"

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
  QAction* newItemAction = fileMenu->addAction("New Item");
  QAction* quitAction = fileMenu->addAction("Quit");



  bench = new Workbench();
  bench->setGeometry(0, 0, 600, 600);

  this->setGeometry(150, 150, 800, 600);

  propertyGrid = new QTreeView();
  propertyGrid->setAllColumnsShowFocus(false);
  propertyGrid->setAlternatingRowColors(true);
  propertyGrid->setSelectionBehavior(QAbstractItemView::SelectItems);
  propertyGrid->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  propertyGrid->setItemDelegate(new PropertyGridDelegate());
  propertyGrid->setItemDelegate(new PropertyGridDelegate());

  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(bench);
  splitter->addWidget(propertyGrid);
  setCentralWidget(splitter);
  centralWidget = splitter;

  connect(newItemAction, SIGNAL(triggered()), this, SLOT(newItem()));
  connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));
  connect(bench, SIGNAL(itemSelected(ToolItem*)), this, SLOT(itemSelected(ToolItem*)));
}

MainWindow::~MainWindow()
{
  delete fileMenu;
  delete centralWidget;
}

void MainWindow::quit() {
  this->close();
}

void MainWindow::newItem() {
  bench->addToolItem(new ToolItem(new Person()));
}

void MainWindow::itemSelected(ToolItem* item) {
  propertyGrid->setModel(item->getModel());
}

}

}
