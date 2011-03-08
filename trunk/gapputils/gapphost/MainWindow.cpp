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

#include <DescriptionAttribute.h>
#include <ReflectableAttribute.h>
#include <ClassProperty.h>
#include <ScalarAttribute.h>

#include "Person.h"

using namespace std;
using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;

namespace gapputils {

namespace host {

void buildModel(QStandardItem* parentItem, const ReflectableClass& object) {
  vector<IClassProperty*> properties = object.getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    QStandardItem *key = new QStandardItem(properties[i]->getName().c_str());
    key->setEditable(false);

    QStandardItem* value = new QStandardItem(properties[i]->getStringValue(object).c_str());
    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description) {
      key->setToolTip(description->getDescription().c_str());
      value->setToolTip(description->getDescription().c_str());
    }

    IReflectableAttribute* reflectable = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* subObject = (ReflectableClass*)properties[i]->getValuePtr(object);
      if (subObject->getAttribute<ScalarAttribute>())
        value->setText(properties[i]->getStringValue(object).c_str());
      else
        value->setText(subObject->getClassName().c_str());
      value->setEnabled(false);
      buildModel(key, *subObject);
    }

    parentItem->setChild(i, 0, key);
    parentItem->setChild(i, 1, value);
  }
}

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
  setWindowTitle("Application Host");
  fileMenu = menuBar()->addMenu("File");
  QAction* quitAction = fileMenu->addAction("Quit");

  connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  QLabel* helloLabel = new QLabel("Hello", this);
  
  QTreeView* tree = new QTreeView();
  tree->setAllColumnsShowFocus(false);
  tree->setAlternatingRowColors(true);
  tree->setSelectionBehavior(QAbstractItemView::SelectItems);
  tree->setEditTriggers(QAbstractItemView::DoubleClicked
                                 | QAbstractItemView::CurrentChanged);
  
  Person person;
  QStandardItemModel* model = new QStandardItemModel(0, 2);
  model->setHorizontalHeaderItem(0, new QStandardItem("Property"));
  model->setHorizontalHeaderItem(1, new QStandardItem("Value"));

  QStandardItem *parentItem = model->invisibleRootItem();
  buildModel(parentItem, person);
  tree->setModel(model);
  
  QComboBox* box1 = new QComboBox();
  box1->addItem("Red");
  box1->addItem("Green");
  //tree->setIndexWidget(model->index(3, 1), box1);

  QSplitter* splitter = new QSplitter(Qt::Horizontal);
  splitter->addWidget(helloLabel);
  splitter->addWidget(tree);
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
