/*
 * GlobalPropertiesView.cpp
 *
 *  Created on: Jul 12, 2012
 *      Author: tombr
 */

#include "GlobalPropertiesView.h"

#include <qboxlayout.h>
#include <qtreewidget.h>

#include <boost/units/detail/utility.hpp>

#include "Workflow.h"
#include "PropertyReference.h"

#include <gapputils/InterfaceAttribute.h>

using namespace gapputils::workflow;
using namespace gapputils::attributes;

namespace gapputils {
namespace host {

GlobalPropertiesView::GlobalPropertiesView(QWidget* parent) :QWidget(parent) {
  propertiesWidget = new QTreeWidget();
  propertiesWidget->setHeaderLabels(QStringList() << "Property" << "Type" << "Label" << "Module" << "Node UUID");

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(propertiesWidget);
  mainLayout->setMargin(0);

  setLayout(mainLayout);
}

GlobalPropertiesView::~GlobalPropertiesView() { }

void GlobalPropertiesView::setWorkflow(boost::shared_ptr<Workflow> workflow) {

  // TODO: observe changes of the workflow and reflect them.
  // TODO: double click jump to module in workbench
  // TODO: Turn the view into an editor (delete global properties, global edges, rename properties)

  this->workflow = workflow;

  propertiesWidget->clear();
  propertiesWidget->setHeaderLabels(QStringList() << "Property" << "Type"  << "Label" << "Module" << "Node UUID");

  std::vector<boost::shared_ptr<GlobalProperty> >& globals = *workflow->getGlobalProperties();
  for (size_t iProp = 0; iProp < globals.size(); ++iProp) {

    QTreeWidgetItem* propItem = new QTreeWidgetItem();

    PropertyReference ref(workflow, globals[iProp]->getModuleUuid(), globals[iProp]->getPropertyId());
    assert(ref.getNode()->getModule());
    std::string label = "<unknown>";
    if (ref.getNode()->getModule()->findProperty("Label"))
      label = ref.getNode()->getModule()->findProperty("Label")->getStringValue(*ref.getNode()->getModule());

    propItem->setText(0, (globals[iProp]->getName() + " (" + globals[iProp]->getPropertyId() + ")").c_str());
    propItem->setText(1, boost::units::detail::demangle(ref.getProperty()->getType().name()).c_str());
    propItem->setText(2, label.c_str());
    propItem->setText(3, ref.getNode()->getModule()->getClassName().c_str());
    propItem->setText(4, globals[iProp]->getModuleUuid().c_str());

    std::vector<boost::weak_ptr<Edge> >& edges = *globals[iProp]->getEdges();
    for (size_t iEdge = 0; iEdge < edges.size(); ++iEdge) {
      boost::shared_ptr<Edge> edge = edges[iEdge].lock();
      QTreeWidgetItem* subItem = new QTreeWidgetItem();

      PropertyReference& ref = *edge->getInputReference();
      assert(ref.getNode()->getModule());
      std::string label = "<unknown>";
      if (ref.getNode()->getModule() && ref.getNode()->getModule()->findProperty("Label"))
        label = ref.getNode()->getModule()->findProperty("Label")->getStringValue(*ref.getNode()->getModule());

      if (ref.getObject()->getAttribute<InterfaceAttribute>()) {
        subItem->setText(0, ref.getObject()->findProperty("Label")->getStringValue(*ref.getObject()).c_str());
      } else {
        subItem->setText(0, ref.getPropertyId().c_str());
      }
      subItem->setText(1, boost::units::detail::demangle(ref.getProperty()->getType().name()).c_str());
      subItem->setText(2, label.c_str());
      subItem->setText(3, ref.getNode()->getModule()->getClassName().c_str());
      subItem->setText(4, ref.getNodeId().c_str());
      propItem->addChild(subItem);
    }
    propertiesWidget->addTopLevelItem(propItem);
  }
}

} /* namespace host */
} /* namespace gapputils */
