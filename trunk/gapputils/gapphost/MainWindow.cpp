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

using namespace std;
using namespace capputils;
using namespace capputils::reflection;
using namespace capputils::attributes;

namespace gapputils {

namespace host {

void buildModel(QStandardItem* parentItem, ReflectableClass& object) {
  vector<IClassProperty*> properties = object.getProperties();

  for (unsigned i = 0; i < properties.size(); ++i) {
    QStandardItem *key = new QStandardItem(properties[i]->getName().c_str());
    QStandardItem* value = new QStandardItem(properties[i]->getStringValue(object).c_str());
    key->setEditable(false);
    value->setData(QVariant::fromValue(PropertyReference(&object, properties[i])), Qt::UserRole);
    
    DescriptionAttribute* description = properties[i]->getAttribute<DescriptionAttribute>();
    if (description) {
      key->setToolTip(description->getDescription().c_str());
      value->setToolTip(description->getDescription().c_str());
    }

    IReflectableAttribute* reflectable = properties[i]->getAttribute<IReflectableAttribute>();
    if (reflectable) {
      ReflectableClass* subObject = reflectable->getValuePtr(object, properties[i]);

      Enumerator* enumerator = dynamic_cast<Enumerator*>(subObject);
      if (enumerator) {
        value->setText(properties[i]->getStringValue(object).c_str());
      } else {
        if (subObject->getAttribute<ScalarAttribute>()) {
          value->setText(properties[i]->getStringValue(object).c_str());
        } else {
          value->setText(subObject->getClassName().c_str());
          value->setEnabled(false);
        }
        buildModel(key, *subObject);
      }
    }
    parentItem->setChild(i, 0, key);
    parentItem->setChild(i, 1, value);
  }
}

void addWidgets(QAbstractItemView* view, QStandardItem* item) {
  for (int i = 0; i < item->rowCount(); ++i) {
    QStandardItem* subItem = item->child(i, 1);
    if (subItem->data(Qt::UserRole).canConvert<PropertyReference>()) {
      const PropertyReference& reference = subItem->data(Qt::UserRole).value<PropertyReference>();
      IReflectableAttribute* reflectable = reference.getProperty()->getAttribute<IReflectableAttribute>();
      if (reflectable) {
        Enumerator* enumerator = dynamic_cast<Enumerator*>(reflectable->getValuePtr(*reference.getObject(), reference.getProperty()));
        if (enumerator) {
          QComboBox* box = new QComboBox();
          vector<string>& values = enumerator->getValues();
          for (unsigned i = 0; i < values.size(); ++i)
            box->addItem(values[i].c_str());
          view->setIndexWidget(subItem->index(), box);
        }
      }
    }
  }
}

MainWindow::MainWindow(QWidget *parent, Qt::WFlags flags)
    : QMainWindow(parent, flags)
{
  setWindowTitle("Application Host");
  fileMenu = menuBar()->addMenu("File");
  QAction* quitAction = fileMenu->addAction("Quit");

  connect(quitAction, SIGNAL(triggered()), this, SLOT(quit()));

  testLabel = new QLabel("Hello", this);
  testLabel->setGeometry(0, 0, 540, 480);
  this->setGeometry(0, 0, 640, 480);
  
  QTreeView* tree = new QTreeView();
  tree->setAllColumnsShowFocus(false);
  tree->setAlternatingRowColors(true);
  tree->setSelectionBehavior(QAbstractItemView::SelectItems);
  tree->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  
  harmonizer1 = new ModelHarmonizer(&person);
  harmonizer2 = new ModelHarmonizer(&person);
  tree->setModel(harmonizer1->getModel());

  QTreeView* tree2 = new QTreeView();
  tree2->setAllColumnsShowFocus(false);
  tree2->setAlternatingRowColors(true);
  tree2->setSelectionBehavior(QAbstractItemView::SelectItems);
  tree2->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  tree2->setModel(harmonizer2->getModel());

//  Person person;
  /*QStandardItemModel* model = new QStandardItemModel(0, 2);
  model->setHorizontalHeaderItem(0, new QStandardItem("Property"));
  model->setHorizontalHeaderItem(1, new QStandardItem("Value"));
  connect(model, SIGNAL(itemChanged(QStandardItem*)), this, SLOT(itemChanged(QStandardItem*)));

  QStandardItem *parentItem = model->invisibleRootItem();
  buildModel(parentItem, person);
  tree->setModel(model);
  addWidgets(tree, parentItem);
  tree->setGeometry(0, 0, 50, 50);*/

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
