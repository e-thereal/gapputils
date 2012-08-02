/*
 * PropertyGrid.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "PropertyGrid.h"

#include <qlabel.h>

#include <capputils/DescriptionAttribute.h>

#include <boost/units/detail/utility.hpp>
#include <qmenu.h>
#include <qmessagebox.h>

#include "GlobalProperty.h"
#include "GlobalEdge.h"
#include "PropertyReference.h"
#include "PropertyGridDelegate.h"
#include "MakeGlobalDialog.h"
#include "PopUpList.h"
#include "Workflow.h"
#include "ModelHarmonizer.h"

using namespace capputils::attributes;
using namespace capputils::reflection;
using namespace gapputils::workflow;

namespace gapputils {

namespace host {

PropertyGrid::PropertyGrid(QWidget* parent) : QSplitter(Qt::Vertical, parent), node(0) {
  propertyGrid = new QTreeView();
  propertyGrid->setAllColumnsShowFocus(false);
  propertyGrid->setAlternatingRowColors(true);
  propertyGrid->setSelectionBehavior(QAbstractItemView::SelectItems);
  propertyGrid->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  propertyGrid->setItemDelegate(new PropertyGridDelegate());

  // Context Menu
  makeGlobal = new QAction("Make Global", propertyGrid);
  removeGlobal = new QAction("Remove Global", propertyGrid);
  connectToGlobal = new QAction("Connect", propertyGrid);
  disconnectFromGlobal = new QAction("Disconnect", propertyGrid);
  connect(makeGlobal, SIGNAL(triggered()), this, SLOT(makePropertyGlobal()));
  connect(removeGlobal, SIGNAL(triggered()), this, SLOT(removePropertyFromGlobal()));
  connect(connectToGlobal, SIGNAL(triggered()), this, SLOT(connectProperty()));
  connect(disconnectFromGlobal, SIGNAL(triggered()), this, SLOT(disconnectProperty()));

  propertyGrid->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(propertyGrid, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
  connect(propertyGrid, SIGNAL(clicked(const QModelIndex&)), this, SLOT(gridClicked(const QModelIndex&)));
  addWidget(propertyGrid);

  QWidget* infoWidget = new QWidget();
  infoLayout = new QFormLayout();
  infoWidget->setLayout(infoLayout);
  addWidget(infoWidget);
}

PropertyGrid::~PropertyGrid() {
}

void PropertyGrid::setEnabled(bool enabled) {
  propertyGrid->setEnabled(enabled);
}

void PropertyGrid::setNode(workflow::Node* node) {
  this->node = node;
  harmonizer = boost::shared_ptr<ModelHarmonizer>(new ModelHarmonizer(node));

  if (node) {
    propertyGrid->setModel(harmonizer->getModel());
    propertyGrid->expandAll();
  } else {
    propertyGrid->setModel(0);
  }
}

QLabel* createTopAlignedLabel(const std::string& text) {
  QLabel* label = new QLabel(text.c_str());
  label->setAlignment(Qt::AlignTop);
  return label;
}

void PropertyGrid::gridClicked(const QModelIndex& index) {
  assert(node);

  const QModelIndex& valueIndex = index.sibling(index.row(), 1);
  if (!valueIndex.isValid())
    return;

  QVariant varient = valueIndex.data(Qt::UserRole);
  if (!varient.canConvert<PropertyReference>())
   return;

  const PropertyReference& reference = varient.value<PropertyReference>();
  IClassProperty* prop = reference.getProperty();

  while (infoLayout->count()) {
    QLayoutItem *item = infoLayout->takeAt(0);
    if (item) {
      delete item->widget();
      delete item;
    } else {
      break;
    }
  }

  QLabel* description;
  DescriptionAttribute* descAttr = prop->getAttribute<DescriptionAttribute>();
  if (descAttr)
    description = new QLabel((std::string("<b>") + descAttr->getDescription() + "</b>").c_str());
  else
    description = new QLabel((std::string("<b>") + prop->getName() + "</b>").c_str());
  description->setWordWrap(true);
  infoLayout->addRow(description);
  QLabel* typeLabel = new QLabel(boost::units::detail::demangle(prop->getType().name()).c_str());
  typeLabel->setWordWrap(true);
  typeLabel->setMinimumSize(10, 10);
  infoLayout->addRow(createTopAlignedLabel("Type:"), typeLabel);

  // check if global and check if connected and fill actions list accordingly
  GlobalProperty* gprop = node->getGlobalProperty(reference);
  if (gprop)
    infoLayout->addRow("Name:", new QLabel(gprop->getName().c_str()));

  GlobalEdge* edge = node->getGlobalEdge(reference);
  if (edge)
    infoLayout->addRow("Connection:", new QLabel(edge->getGlobalProperty().c_str()));

  if (node) {
    QLabel* uuidLabel = new QLabel(node->getUuid().c_str());
    uuidLabel->setMinimumSize(10, 10);
    infoLayout->addRow("Uuid:", uuidLabel);
  }

  if (reference.getObject()) {
    QLabel* moduleTypeLabel = new QLabel(reference.getObject()->getClassName().c_str());
    moduleTypeLabel->setWordWrap(true);
    moduleTypeLabel->setMinimumSize(10, 10);
    infoLayout->addRow(createTopAlignedLabel("Module:"), moduleTypeLabel);
  }
}

void PropertyGrid::showContextMenu(const QPoint& point) {
  QList<QAction*> actions;
  QModelIndex index = propertyGrid->indexAt(point);
  if (!index.isValid())
    return;

  QVariant varient = index.data(Qt::UserRole);
  if (!varient.canConvert<PropertyReference>())
    return;

  const PropertyReference& reference = varient.value<PropertyReference>();

  // check if global and check if connected and fill actions list accordingly
  GlobalProperty* gprop = node->getGlobalProperty(reference);
  if (gprop)
    actions.append(removeGlobal);
  else
    actions.append(makeGlobal);

  GlobalEdge* edge = node->getGlobalEdge(reference);
  if (edge)
    actions.append(disconnectFromGlobal);
  else
    actions.append(connectToGlobal);

  QMenu::exec(actions, propertyGrid->mapToGlobal(point));
}

void PropertyGrid::makePropertyGlobal() {
  assert(node);
  assert(node->getWorkflow());
  MakeGlobalDialog dialog(propertyGrid);
  if (dialog.exec() == QDialog::Accepted) {
    QString text = dialog.getText();
    if (text.length()) {
      QModelIndex index = propertyGrid->currentIndex();

      QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
      if (model) {
        QStandardItem* item = model->itemFromIndex(index);
        QFont font = item->font();
        font.setUnderline(true);
        item->setFont(font);
      }

      node->getWorkflow()->makePropertyGlobal(text.toAscii().data(), index.data(Qt::UserRole).value<PropertyReference>());
    } else {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for a global property!");
    }
  }
}

void PropertyGrid::removePropertyFromGlobal() {
  assert(node);
  assert(node->getWorkflow());
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
  if (model) {
    QStandardItem* item = model->itemFromIndex(index);
    QFont font = item->font();
    font.setUnderline(false);
    item->setFont(font);
  }

  GlobalProperty* gprop = node->getGlobalProperty(reference);
  if (gprop)
    node->getWorkflow()->removeGlobalProperty(gprop);
}

void PropertyGrid::connectProperty() {
  assert(node);
  assert(node->getWorkflow());
  workflow::Workflow* workflow = node->getWorkflow();

  // Get list of compatible global properties.
  // Show list windows.
  // establish connection.
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  PopUpList list;
  std::vector<GlobalProperty*>* globals = workflow->getGlobalProperties();
  for (unsigned i = 0; i < globals->size(); ++i) {
    PropertyReference ref(reference.getWorkflow(), globals->at(i)->getModuleUuid(), globals->at(i)->getPropertyId());
    if (Edge::areCompatible(ref.getProperty(), reference.getProperty())) {
      list.getList()->addItem(globals->at(i)->getName().c_str());
    } else {
//      GlobalProperty* gprop = globals->at(i);
      //cout << gprop->getName() << " is not compatible." << endl;
      //cout << gprop->getProperty()->getType().name() << " != " << reference.getProperty()->getType().name() << endl;
    }
  }
  if (list.getList()->count() == 0) {
    QMessageBox::information(0, "No Compatible Global Connection", "There are no compatible global connections to connect to.");
    return;
  }

  QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
  if (model) {
    QStandardItem* item = model->itemFromIndex(index);
    QFont font = item->font();
    font.setItalic(true);
    item->setFont(font);
  }

  if (list.exec() == QDialog::Accepted) {
    workflow->connectProperty(list.getList()->selectedItems()[0]->text().toAscii().data(), reference);
  }
}

void PropertyGrid::disconnectProperty() {
  assert(node);
  assert(node->getWorkflow());
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
  if (model) {
    QStandardItem* item = model->itemFromIndex(index);
    QFont font = item->font();
    font.setItalic(false);
    item->setFont(font);
  }

  GlobalEdge* edge = node->getGlobalEdge(reference);
  if (edge)
    node->getWorkflow()->removeGlobalEdge(edge);
}

} /* namespace host */

} /* namespace gapputils */
