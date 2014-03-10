/*
 * GlobalPropertiesView.cpp
 *
 *  Created on: Jul 12, 2012
 *      Author: tombr
 */

#include "GlobalPropertiesView.h"

#include <qboxlayout.h>
#include <qtreewidget.h>
#include <qsettings.h>
#include <qevent.h>
#include <qaction.h>
#include <qkeysequence.h>

#include <boost/units/detail/utility.hpp>

#include "Workflow.h"
#include "PropertyReference.h"
#include "GlobalPropertiesViewDelegate.h"

#include <gapputils/attributes/InterfaceAttribute.h>

using namespace gapputils::workflow;
using namespace gapputils::attributes;

namespace gapputils {

namespace host {

enum TableColumns {PropertyColumn, TypeColumn, ModuleColumn, UuidColumn};

GlobalPropertiesView::GlobalPropertiesView(QWidget* parent) : QWidget(parent), eventHandler(this, &GlobalPropertiesView::handleChanged) {
  propertiesWidget = new QTreeWidget();
  propertiesWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
  propertiesWidget->setHeaderLabels(QStringList() << "Property Connections" << "Type" << "Module" << "Node UUID");
  propertiesWidget->setContextMenuPolicy(Qt::ActionsContextMenu);
  propertiesWidget->setItemDelegate(new GlobalPropertiesViewDelegate());

  QAction* action = new QAction("Edit Property Name", this);
  action->setShortcut(QKeySequence(Qt::Key_F2));
  propertiesWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(editPropertyName()));

  action = new QAction("Delete", this);
  propertiesWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(deletePropertyOrEdge()));

  connect(propertiesWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)),
      this, SLOT(handleItemDoubleClicked(QTreeWidgetItem*, int)));

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(propertiesWidget);
  mainLayout->setMargin(0);

  setLayout(mainLayout);

  QSettings settings;
  if (settings.contains("globalpropertiesview/PropertyWidth"))
    propertiesWidget->setColumnWidth(PropertyColumn, settings.value("globalpropertiesview/PropertyWidth").toInt());
  if (settings.contains("globalpropertiesview/TypeWidth"))
    propertiesWidget->setColumnWidth(TypeColumn, settings.value("globalpropertiesview/TypeWidth").toInt());
  if (settings.contains("globalpropertiesview/ModuleWidth"))
    propertiesWidget->setColumnWidth(ModuleColumn, settings.value("globalpropertiesview/ModuleWidth").toInt());
}

GlobalPropertiesView::~GlobalPropertiesView() {
  QSettings settings;
  settings.setValue("globalpropertiesview/PropertyWidth", propertiesWidget->columnWidth(PropertyColumn));
  settings.setValue("globalpropertiesview/TypeWidth", propertiesWidget->columnWidth(TypeColumn));
  settings.setValue("globalpropertiesview/ModuleWidth", propertiesWidget->columnWidth(ModuleColumn));
  settings.setValue("globalpropertiesview/UuidWidth", propertiesWidget->columnWidth(UuidColumn));
}

void GlobalPropertiesView::setWorkflow(boost::shared_ptr<Workflow> workflow) {
  boost::shared_ptr<Workflow> oldWorkflow = this->workflow.lock();

  if (!workflow || oldWorkflow == workflow)
    return;

  if (oldWorkflow) {
    oldWorkflow->Changed.disconnect(eventHandler);
  }

  workflow->Changed.connect(eventHandler);

  this->workflow = workflow;
  updateProperties();
}

void GlobalPropertiesView::updateProperties() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  propertiesWidget->clear();
  propertiesWidget->setHeaderLabels(QStringList() << "Property Connections" << "Type" << "Module" << "Node UUID");

  std::vector<boost::shared_ptr<GlobalProperty> >& globals = *workflow->getGlobalProperties();
  for (size_t iProp = 0; iProp < globals.size(); ++iProp) {

    QTreeWidgetItem* propItem = new QTreeWidgetItem();
    propItem->setFlags(propItem->flags() | Qt::ItemIsEditable);

    PropertyReference ref(workflow, globals[iProp]->getModuleUuid(), globals[iProp]->getPropertyId());
    assert(ref.getNode()->getModule());
    std::string label = "<unknown>";
    if (ref.getNode()->getModule()->findProperty("Label"))
      label = ref.getNode()->getModule()->findProperty("Label")->getStringValue(*ref.getNode()->getModule());

    propItem->setText(0, (globals[iProp]->getName() + " (" + label + "::" + ref.getPropertyId() + ")").c_str());
    propItem->setText(1, boost::units::detail::demangle(ref.getProperty()->getType().name()).c_str());
    propItem->setText(2, ref.getNode()->getModule()->getClassName().c_str());
    propItem->setText(3, globals[iProp]->getModuleUuid().c_str());

    propItem->setData(0, Qt::UserRole, QVariant::fromValue(ref));

    std::vector<boost::weak_ptr<GlobalEdge> >& edges = *globals[iProp]->getEdges();
    for (size_t iEdge = 0; iEdge < edges.size(); ++iEdge) {
      boost::shared_ptr<GlobalEdge> edge = edges[iEdge].lock();
      QTreeWidgetItem* subItem = new QTreeWidgetItem();

      PropertyReference& ref = *edge->getInputReference();
      assert(ref.getNode()->getModule());
      std::string label = "<unknown>";
      if (ref.getNode()->getModule() && ref.getNode()->getModule()->findProperty("Label"))
        label = ref.getNode()->getModule()->findProperty("Label")->getStringValue(*ref.getNode()->getModule());

      subItem->setText(0, (label + "::" + ref.getPropertyId()).c_str());
      subItem->setText(1, boost::units::detail::demangle(ref.getProperty()->getType().name()).c_str());
      subItem->setText(2, ref.getNode()->getModule()->getClassName().c_str());
      subItem->setText(3, ref.getNodeId().c_str());
      subItem->setData(0, Qt::UserRole, QVariant::fromValue(ref));

      propItem->addChild(subItem);
    }

    std::vector<boost::weak_ptr<Expression> >& expressions = *globals[iProp]->getExpressions();
    for (size_t iExpr = 0; iExpr < expressions.size(); ++iExpr) {
      boost::shared_ptr<Expression> expression = expressions[iExpr].lock();
      boost::shared_ptr<Node> node = expression->getNode().lock();
      boost::shared_ptr<Workflow> workflow = node->getWorkflow().lock();
      QTreeWidgetItem* subItem = new QTreeWidgetItem();

      PropertyReference ref(workflow, node->getUuid(), expression->getPropertyName());
      assert(ref.getNode()->getModule());
      std::string label = "<unknown>";
      if (node->getModule() && node->getModule()->findProperty("Label"))
        label = node->getModule()->findProperty("Label")->getStringValue(*node->getModule());

      subItem->setText(0, (label + "::" + ref.getPropertyId()).c_str());
      subItem->setText(1, boost::units::detail::demangle(ref.getProperty()->getType().name()).c_str());
      subItem->setText(2, ref.getNode()->getModule()->getClassName().c_str());
      subItem->setText(3, ref.getNodeId().c_str());
      subItem->setData(0, Qt::UserRole, QVariant::fromValue(ref));

      QFont font = subItem->font(0);
      font.setItalic(true);
      subItem->setFont(0, font);

      propItem->addChild(subItem);
    }

    propertiesWidget->addTopLevelItem(propItem);
  }
}

void GlobalPropertiesView::handleChanged(capputils::ObservableClass* /*object*/, int eventId) {
  if (eventId == Workflow::globalPropertiesId || eventId == Workflow::globalEdgesId) {
    updateProperties();
  }
}

void GlobalPropertiesView::deletePropertyOrEdge() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  if (!propertiesWidget->currentItem()->data(0, Qt::UserRole).canConvert<PropertyReference>())
    return;

  PropertyReference reference = propertiesWidget->currentItem()->data(0, Qt::UserRole).value<PropertyReference>();
  if (propertiesWidget->currentIndex().parent().isValid()) {
    // remove edge
    boost::shared_ptr<GlobalEdge> edge = workflow->getGlobalEdge(reference);
    if (edge)
      workflow->removeGlobalEdge(edge);
  } else {
    // remove property
    boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
    if (gprop)
      workflow->removeGlobalProperty(gprop);
  }
}

void GlobalPropertiesView::editPropertyName() {
  boost::shared_ptr<Workflow> workflow = this->workflow.lock();

  if (!propertiesWidget->currentItem() || !propertiesWidget->currentItem()->data(0, Qt::UserRole).canConvert<PropertyReference>())
    return;

  PropertyReference reference = propertiesWidget->currentItem()->data(0, Qt::UserRole).value<PropertyReference>();
  if (!propertiesWidget->currentIndex().parent().isValid()) {
    boost::shared_ptr<GlobalProperty> gprop = workflow->getGlobalProperty(reference);
    if (propertiesWidget->currentItem() && gprop) {
      propertiesWidget->editItem(propertiesWidget->currentItem());
    }
  }
}

void GlobalPropertiesView::keyPressEvent(QKeyEvent* event) {
  switch (event->key()) {
  case Qt::Key_Delete:
    deletePropertyOrEdge();
    break;
  default:
    QWidget::keyPressEvent(event);
  }
}

void GlobalPropertiesView::handleItemDoubleClicked(QTreeWidgetItem* item, int /*column*/) {
  Q_EMIT selectModuleRequested(item->text(UuidColumn));
}

} /* namespace host */

} /* namespace gapputils */
