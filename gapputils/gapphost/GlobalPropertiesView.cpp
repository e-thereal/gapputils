/*
 * GlobalPropertiesView.cpp
 *
 *  Created on: Jul 12, 2012
 *      Author: tombr
 */

#include "GlobalPropertiesView.h"

#include <qboxlayout.h>
#include <qtreewidget.h>

namespace gapputils {
namespace host {

GlobalPropertiesView::GlobalPropertiesView(QWidget* parent) :QWidget(parent) {
  QTreeWidget* propertiesWidget = new QTreeWidget();
  propertiesWidget->setHeaderLabels(QStringList() << "Property" << "Type" << "Module" << "Node UUID");

  QTreeWidgetItem* subItem = new QTreeWidgetItem();
  subItem->setText(0, "Width");
  subItem->setText(1, "int");
  subItem->setText(2, "gapputils::Reader");
  subItem->setText(3, "0F-34-221-34");

  QTreeWidgetItem* propItem = new QTreeWidgetItem();
  propItem->setText(0, "InputWidth (Width)");
  propItem->setText(1, "int");
  propItem->setText(2, "gapputils::Writer");
  propItem->setText(3, "0F-34-221-34");
  propItem->addChild(subItem);
  propertiesWidget->addTopLevelItem(propItem);

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(propertiesWidget);
  mainLayout->setMargin(0);

  setLayout(mainLayout);
}

GlobalPropertiesView::~GlobalPropertiesView() { }

} /* namespace host */
} /* namespace gapputils */
