/*
 * LogbookWidget.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "LogbookWidget.h"

#include "LogbookModel.h"

namespace gapputils {
namespace host {

LogbookWidget::LogbookWidget(QWidget* parent) : QTreeWidget(parent) {
  setHeaderLabels(QStringList() << "Message" << "Module");

  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, "Hello world");
  item->setText(1, "test");
  this->addTopLevelItem(item);

  LogbookModel& model = LogbookModel::GetInstance();
  connect(&model, SIGNAL(newMessage(const std::string&, const std::string&)), this, SLOT(showMessage(const std::string&, const std::string&)));
}

LogbookWidget::~LogbookWidget() {
}

void LogbookWidget::showMessage(const std::string& message, const std::string& module) {
  QTreeWidgetItem* item = new QTreeWidgetItem();
  item->setText(0, message.c_str());
  item->setText(1, module.c_str());
  this->addTopLevelItem(item);
}

} /* namespace host */

} /* namespace gapputils */
