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

#include <boost/units/detail/utility.hpp>

#include "Workflow.h"
#include "PropertyReference.h"

#include <gapputils/InterfaceAttribute.h>

using namespace gapputils::workflow;
using namespace gapputils::attributes;

namespace gapputils {

namespace host {

enum TableColumns {PropertyColumn, TypeColumn, ModuleColumn, UuidColumn};

GlobalPropertiesView::GlobalPropertiesView(QWidget* parent) :QWidget(parent) {
  propertiesWidget = new QTreeWidget();
  propertiesWidget->setEditTriggers(QAbstractItemView::SelectedClicked);
  propertiesWidget->setHeaderLabels(QStringList() << "Property Connections" << "Type" << "Module" << "Node UUID");

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

  // TODO: observe changes of the workflow and reflect them.
  // TODO: Turn the view into an editor (delete global properties, global edges, rename properties)

  this->workflow = workflow;

  propertiesWidget->clear();
  propertiesWidget->setHeaderLabels(QStringList() << "Property Connections" << "Type" << "Module" << "Node UUID");

  std::vector<boost::shared_ptr<GlobalProperty> >& globals = *workflow->getGlobalProperties();
  for (size_t iProp = 0; iProp < globals.size(); ++iProp) {

    QTreeWidgetItem* propItem = new QTreeWidgetItem();

    PropertyReference ref(workflow, globals[iProp]->getModuleUuid(), globals[iProp]->getPropertyId());
    assert(ref.getNode()->getModule());
    std::string label = "<unknown>";
    if (ref.getNode()->getModule()->findProperty("Label"))
      label = ref.getNode()->getModule()->findProperty("Label")->getStringValue(*ref.getNode()->getModule());

    propItem->setText(0, (globals[iProp]->getName() + " (" + label + "::" + ref.getPropertyId() + ")").c_str());
    propItem->setText(1, boost::units::detail::demangle(ref.getProperty()->getType().name()).c_str());
    propItem->setText(2, ref.getNode()->getModule()->getClassName().c_str());
    propItem->setText(3, globals[iProp]->getModuleUuid().c_str());

    std::vector<boost::weak_ptr<Edge> >& edges = *globals[iProp]->getEdges();
    for (size_t iEdge = 0; iEdge < edges.size(); ++iEdge) {
      boost::shared_ptr<Edge> edge = edges[iEdge].lock();
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
      propItem->addChild(subItem);
    }
    propertiesWidget->addTopLevelItem(propItem);
  }
}

void GlobalPropertiesView::handleItemDoubleClicked(QTreeWidgetItem* item, int /*column*/) {
  Q_EMIT selectModuleRequested(item->text(UuidColumn));
}

} /* namespace host */
} /* namespace gapputils */
