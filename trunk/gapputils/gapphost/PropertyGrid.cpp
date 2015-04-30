/*
 * PropertyGrid.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "PropertyGrid.h"

#include <qlabel.h>

#include <capputils/attributes/DescriptionAttribute.h>
#include <capputils/attributes/InputAttribute.h>
#include <capputils/attributes/OutputAttribute.h>

#include <capputils/reflection/ReflectableClassFactory.h>

#include <boost/units/detail/utility.hpp>
#include <qmenu.h>
#include <qmessagebox.h>
#include <qsettings.h>

#include <cstring>

#include "GlobalProperty.h"
#include "GlobalEdge.h"
#include "PropertyReference.h"
#include "PropertyGridDelegate.h"
#include "LineEditDialog.h"
#include "PopUpList.h"
#include "Workflow.h"
#include "ModelHarmonizer.h"
#include "DataModel.h"
#include "WorkbenchWindow.h"
#include "MainWindow.h"

#include <gapputils/WorkflowElement.h>

using namespace capputils::attributes;
using namespace capputils::reflection;
using namespace gapputils::workflow;

namespace gapputils {

namespace host {

enum PropertyGridColumns {PropertyColumn, ValueColumn};

PropertyGrid::PropertyGrid(QWidget* parent) : QSplitter(Qt::Vertical, parent) {
  propertyGrid = new QTreeView();
  propertyGrid->setAllColumnsShowFocus(false);
  propertyGrid->setAlternatingRowColors(true);
  propertyGrid->setSelectionBehavior(QAbstractItemView::SelectItems);
  propertyGrid->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::CurrentChanged);
  propertyGrid->setItemDelegate(new PropertyGridDelegate());

  propertyGrid->setDefaultDropAction(Qt::MoveAction);
  propertyGrid->setDragEnabled(true);
  propertyGrid->setAcceptDrops(false);
  propertyGrid->setDragDropMode(QAbstractItemView::InternalMove);
  propertyGrid->setDropIndicatorShown(true);
  propertyGrid->setDragDropOverwriteMode(false);

//  propertyGrid->setStyleSheet( "QTreeView::branch {  border-image: url(none.png); }" );

  connect(propertyGrid, SIGNAL(doubleClicked(const QModelIndex&)), this, SLOT(handleDoubleClicked(const QModelIndex&)));

  QStandardItemModel* model = new QStandardItemModel(0, 2);
  model->setHorizontalHeaderItem(0, new QStandardItem("Property"));
  model->setHorizontalHeaderItem(1, new QStandardItem("Value"));
  propertyGrid->setModel(model);

  QSettings settings;
  if (settings.contains("propertygrid/PropertyWidth"))
    propertyGrid->setColumnWidth(PropertyColumn, settings.value("propertygrid/PropertyWidth").toInt());
//  if (settings.contains("propertygrid/ValueWidth"))
//    propertyGrid->setColumnWidth(ValueColumn, settings.value("propertygrid/ValueWidth").toInt());

  // Context Menu
  makeGlobal = new QAction("Make Global", propertyGrid);
  removeGlobal = new QAction("Remove Global", propertyGrid);
  connectToGlobal = new QAction("Connect", propertyGrid);
  disconnectFromGlobal = new QAction("Disconnect", propertyGrid);
  makeParameter = new QAction("Make Parameter", propertyGrid);
  makeInput = new QAction("Make Input", propertyGrid);
  makeOutput = new QAction("Make Output", propertyGrid);
  connect(makeGlobal, SIGNAL(triggered()), this, SLOT(makePropertyGlobal()));
  connect(removeGlobal, SIGNAL(triggered()), this, SLOT(removePropertyFromGlobal()));
  connect(connectToGlobal, SIGNAL(triggered()), this, SLOT(connectProperty()));
  connect(disconnectFromGlobal, SIGNAL(triggered()), this, SLOT(disconnectProperty()));
  connect(makeParameter, SIGNAL(triggered()), this, SLOT(makePropertyParameter()));
  connect(makeInput, SIGNAL(triggered()), this, SLOT(makePropertyInput()));
  connect(makeOutput, SIGNAL(triggered()), this, SLOT(makePropertyOutput()));

  propertyGrid->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(propertyGrid, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
//  connect(propertyGrid, SIGNAL(clicked(const QModelIndex&)), this, SLOT(gridClicked(const QModelIndex&)));
  addWidget(propertyGrid);

  QWidget* infoWidget = new QWidget();
  infoLayout = new QFormLayout();
  infoWidget->setLayout(infoLayout);
  addWidget(infoWidget);
}

PropertyGrid::~PropertyGrid() {
  QSettings settings;
  settings.setValue("propertygrid/PropertyWidth", propertyGrid->columnWidth(PropertyColumn));
  settings.setValue("propertygrid/ValueWidth", propertyGrid->columnWidth(ValueColumn));
}

void PropertyGrid::setEnabled(bool enabled) {
  propertyGrid->setEnabled(enabled);
}

void PropertyGrid::setNode(boost::shared_ptr<workflow::Node> node) {
  this->node = node;

  if (node) {
    harmonizer = boost::shared_ptr<ModelHarmonizer>(new ModelHarmonizer(node));
    propertyGrid->setModel(harmonizer->getModel());
    connect(propertyGrid->selectionModel(), SIGNAL(currentChanged(const QModelIndex&, const QModelIndex&)), this, SLOT(currentChanged(const QModelIndex&, const QModelIndex&)));
    propertyGrid->expandAll();
    propertyGrid->setCurrentIndex(propertyGrid->model()->index(0, 0));

    for (int iRow = 0; iRow < propertyGrid->model()->rowCount(); ++iRow) {
      if (!harmonizer->getModel()->invisibleRootItem()->child(iRow, 1))
        propertyGrid->setFirstColumnSpanned(iRow, harmonizer->getModel()->invisibleRootItem()->index(), true);
    }
  } else {
    propertyGrid->setModel(0);
  }
}

QLabel* createTopAlignedLabel(const std::string& text) {
  QLabel* label = new QLabel(text.c_str());
  label->setAlignment(Qt::AlignTop);
  return label;
}

void PropertyGrid::currentChanged(const QModelIndex& current, const QModelIndex&) {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();
  assert(node);

  const QModelIndex& valueIndex = current.sibling(current.row(), 1);
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
  boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
  if (gprop)
    infoLayout->addRow("Name:", new QLabel(gprop->getName().c_str()));

  boost::shared_ptr<GlobalEdge> edge = workflow->getGlobalEdge(reference);
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

void PropertyGrid::handleDoubleClicked(const QModelIndex& index) {
  if (index.column() != 0)
    return;

  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  assert(node);
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = boost::dynamic_pointer_cast<gapputils::workflow::Workflow>(node);
  if (!workflow)
    return;

  const QModelIndex& valueIndex = index.sibling(index.row(), 1);
  if (!valueIndex.isValid())
    return;

  QVariant varient = valueIndex.data(Qt::UserRole);
  if (!varient.canConvert<PropertyReference>())
   return;

  const PropertyReference& reference = varient.value<PropertyReference>();

  if (workflow->getNode(reference.getPropertyId())) {
    Q_EMIT selectModuleRequested(reference.getPropertyId().c_str());
  }
}

void PropertyGrid::showContextMenu(const QPoint& point) {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();

  QList<QAction*> actions;
  QModelIndex index = propertyGrid->indexAt(point);
  if (!index.isValid())
    return;

  propertyGrid->closePersistentEditor(index);

  QVariant varient = index.data(Qt::UserRole);
  if (!varient.canConvert<PropertyReference>())
    return;

  const PropertyReference& reference = varient.value<PropertyReference>();

  // check if global and check if connected and fill actions list accordingly
  boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
  if (gprop)
    actions.append(removeGlobal);
  else
    actions.append(makeGlobal);

  boost::shared_ptr<GlobalEdge> edge = workflow->getGlobalEdge(reference);
  if (edge) {
    actions.append(disconnectFromGlobal);
  } else {
    actions.append(connectToGlobal);
    actions.append(makeParameter);
    if (reference.getProperty()->getAttribute<InputAttribute>()) {
      if (boost::dynamic_pointer_cast<Workflow>(reference.getNode()))
        actions.append(makeOutput);
      else
        actions.append(makeInput);
    }
    if (reference.getProperty()->getAttribute<OutputAttribute>()) {
      if (boost::dynamic_pointer_cast<Workflow>(reference.getNode()))
        actions.append(makeInput);
      else
        actions.append(makeOutput);
    }
  }

  QMenu::exec(actions, propertyGrid->mapToGlobal(point));
}

void PropertyGrid::makePropertyGlobal() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();

  LineEditDialog dialog("Enter the name of the global property:", propertyGrid);
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

      workflow->makePropertyGlobal(text.toAscii().data(), index.data(Qt::UserRole).value<PropertyReference>());
    } else {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for a global property!");
    }
  }
}

void PropertyGrid::removePropertyFromGlobal() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();

  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
  if (model) {
    QStandardItem* item = model->itemFromIndex(index);
    QFont font = item->font();
    font.setUnderline(false);
    item->setFont(font);
  }

  boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
  if (gprop)
    workflow->removeGlobalProperty(gprop);
}

void PropertyGrid::connectProperty() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();

  // Get list of compatible global properties.
  // Show list windows.
  // establish connection.
  QModelIndex index = propertyGrid->currentIndex();

  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  PopUpList list("Select the global property:", propertyGrid);
  boost::shared_ptr<std::vector<boost::shared_ptr<GlobalProperty> > > globals = workflow->getGlobalProperties();
  for (unsigned i = 0; i < globals->size(); ++i) {
    PropertyReference ref(reference.getWorkflow(), globals->at(i)->getModuleUuid(), globals->at(i)->getPropertyId());
    if (Edge::areCompatible(ref.getProperty(), reference.getProperty())) {
      list.getList()->addItem(globals->at(i)->getName().c_str());
    }
  }
  if (list.getList()->count() == 0) {
    QMessageBox::information(0, "No Compatible Global Connection", "There are no compatible global connections to connect to.");
    return;
  }

  if (list.exec() == QDialog::Accepted) {
    workflow->connectProperty(list.getList()->selectedItems()[0]->text().toAscii().data(), reference);
    QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
    if (model) {
      QStandardItem* item = model->itemFromIndex(index);
      QFont font = item->font();
      font.setItalic(true);
      item->setFont(font);
    }
  }
}

void PropertyGrid::disconnectProperty() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();
  
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  QStandardItemModel* model = dynamic_cast<QStandardItemModel*>(propertyGrid->model());
  if (model) {
    QStandardItem* item = model->itemFromIndex(index);
    QFont font = item->font();
    font.setItalic(false);
    item->setFont(font);
  }

  boost::shared_ptr<GlobalEdge> edge = workflow->getGlobalEdge(reference);
  if (edge)
    workflow->removeGlobalEdge(edge);
}

void PropertyGrid::makePropertyParameter() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();

  // Get compatible parameter module
  // Create parameter module
  // Make its value property global
  // Connect to newly created global property
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  std::vector<std::string>& classnames = factory.getClassNames();

  std::string classname;
  for (size_t i = 0; i < classnames.size(); ++i) {
    if (strncmp(classnames[i].c_str(), "interfaces::parameters", strlen("interfaces::parameters")) == 0) {
      boost::shared_ptr<ReflectableClass> object(factory.newInstance(classnames[i]));
      IClassProperty* valueProp = object->findProperty("Value");
      if (!object->findProperty("Values") && valueProp && valueProp->getType() == reference.getProperty()->getType()) {
        classname = classnames[i];
        break;
      }
    }
  }

  if (classname.size() == 0) {
    QMessageBox::information(0, "No Compatible Parameter Modules", "There are no compatible parameters modules for the selected property.");
    return;
  }

  std::string parameterName;
  LineEditDialog dialog("Enter the name of the parameter:", propertyGrid);
  if (dialog.exec() == QDialog::Accepted) {
    parameterName = dialog.getText().toAscii().data();
    if (parameterName.size() == 0) {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for a parameter!");
      return;
    }
  } else {
    return;
  }

  boost::shared_ptr<Node> parameterNode = DataModel::getInstance().getMainWindow()->getCurrentWorkbenchWindow()->createModule(0, 0, classname.c_str());
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(parameterNode->getModule().get());
  assert(element);

  element->setLabel(parameterName);
  PropertyReference paraRef(workflow, parameterNode->getUuid(), "Value");
  paraRef.getProperty()->setValue(*paraRef.getObject(), *reference.getObject(), reference.getProperty());
  workflow->makePropertyGlobal(parameterName, paraRef);
  workflow->connectProperty(parameterName, reference);
}

void PropertyGrid::makePropertyInput() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();
  std::string connectPropertyName; // name of the property that will be connected

  // Get compatible input module
  // Create input module
  // Connect to newly created input module (create edge)
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  std::vector<std::string>& classnames = factory.getClassNames();

  std::string classname;
  for (size_t i = 0; i < classnames.size(); ++i) {
    if (strncmp(classnames[i].c_str(), "interfaces::inputs", strlen("interfaces::inputs")) == 0) {
      boost::shared_ptr<ReflectableClass> object(factory.newInstance(classnames[i]));
      IClassProperty* valuesProp = object->findProperty("Values");
      if (valuesProp) {
        if (valuesProp->getType() == reference.getProperty()->getType()) {
          classname = classnames[i];
          connectPropertyName = "Values";
          break;
        }
      } else {
        IClassProperty* valueProp = object->findProperty("Value");
        if (valueProp && valueProp->getType() == reference.getProperty()->getType()) {
          classname = classnames[i];
          connectPropertyName = "Value";
          break;
        }
      }
    }
  }

  if (classname.size() == 0) {
    QMessageBox::information(0, "No Compatible Input Modules", "There are no compatible input modules for the selected property.");
    return;
  }

  std::string inputName;
  LineEditDialog dialog("Enter the name of the input module:", propertyGrid);
  if (dialog.exec() == QDialog::Accepted) {
    inputName = dialog.getText().toAscii().data();
    if (inputName.size() == 0) {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for an input module!");
      return;
    }
  } else {
    return;
  }

  boost::shared_ptr<Node> inputNode = DataModel::getInstance().getMainWindow()->getCurrentWorkbenchWindow()->createModule(reference.getNode()->getX() - 160, reference.getNode()->getY(), classname.c_str());
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(inputNode->getModule().get());
  assert(element);

  element->setLabel(inputName);

  PropertyReference paraRef(workflow, inputNode->getUuid(), connectPropertyName);
  DataModel::getInstance().getMainWindow()->getCurrentWorkbenchWindow()->createCable(workflow->createEdge(paraRef, reference));
}

void PropertyGrid::makePropertyOutput() {
  boost::shared_ptr<gapputils::workflow::Node> node = this->node.lock();
  boost::shared_ptr<gapputils::workflow::Workflow> workflow = node->getWorkflow().lock();
  std::string connectPropertyName; // name of the property that will be connected

  // Get compatible input module
  // Create input module
  // Connect to newly created input module (create edge)
  QModelIndex index = propertyGrid->currentIndex();
  PropertyReference reference = index.data(Qt::UserRole).value<PropertyReference>();

  ReflectableClassFactory& factory = ReflectableClassFactory::getInstance();
  std::vector<std::string>& classnames = factory.getClassNames();

  std::string classname;
  for (size_t i = 0; i < classnames.size(); ++i) {
    if (strncmp(classnames[i].c_str(), "interfaces::outputs", strlen("interfaces::outputs")) == 0) {
      boost::shared_ptr<ReflectableClass> object(factory.newInstance(classnames[i]));
      IClassProperty* valuesProp = object->findProperty("Values");
      if (valuesProp) {
        if (valuesProp->getType() == reference.getProperty()->getType()) {
          classname = classnames[i];
          connectPropertyName = "Values";
          break;
        }
      } else {
        IClassProperty* valueProp = object->findProperty("Value");
        if (valueProp && valueProp->getType() == reference.getProperty()->getType()) {
          classname = classnames[i];
          connectPropertyName = "Value";
          break;
        }
      }
    }
  }

  if (classname.size() == 0) {
    QMessageBox::information(0, "No Compatible Input Modules", "There are no compatible input modules for the selected property.");
    return;
  }

  std::string outputName;
  LineEditDialog dialog("Enter the name of the output module:", propertyGrid);
  if (dialog.exec() == QDialog::Accepted) {
    outputName = dialog.getText().toAscii().data();
    if (outputName.size() == 0) {
      QMessageBox::warning(0, "Invalid Name", "The name you have entered is not a valid name for an output module!");
      return;
    }
  } else {
    return;
  }

  boost::shared_ptr<Node> outputNode = DataModel::getInstance().getMainWindow()->getCurrentWorkbenchWindow()->createModule(reference.getNode()->getX() + 160, reference.getNode()->getY(), classname.c_str());
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(outputNode->getModule().get());
  assert(element);

  element->setLabel(outputName);

  PropertyReference paraRef(workflow, outputNode->getUuid(), connectPropertyName);
  DataModel::getInstance().getMainWindow()->getCurrentWorkbenchWindow()->createCable(workflow->createEdge(reference, paraRef));
}

} /* namespace host */

} /* namespace gapputils */
