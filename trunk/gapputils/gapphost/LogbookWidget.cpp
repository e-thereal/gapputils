/*
 * LogbookWidget.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "LogbookWidget.h"

#include "LogbookModel.h"
#include <qicon.h>
#include <qboxlayout.h>
#include <qlabel.h>
#include <qaction.h>

namespace gapputils {
namespace host {

LogbookWidget::LogbookWidget(QWidget* parent)
 : QWidget(parent), traceIcon(":/icons/trace.png"), infoIcon(":/icons/info.png"),
   warningIcon(":/icons/warning.png"), errorIcon(":/icons/error.png")
{
  QLabel* label = new QLabel("Quick filter");
  traceButton = new QToolButton();
  traceButton->setIcon(traceIcon);
  traceButton->setIconSize(QSize(16, 16));
  traceButton->setCheckable(true);
  connect(traceButton, SIGNAL(toggled(bool)), this, SLOT(handleButtonToggle(bool)));

  infoButton = new QToolButton();
  infoButton->setIcon(infoIcon);
  infoButton->setIconSize(QSize(16, 16));
  infoButton->setCheckable(true);
  infoButton->setChecked(true);
  connect(infoButton, SIGNAL(toggled(bool)), this, SLOT(handleButtonToggle(bool)));

  warningButton = new QToolButton();
  warningButton->setIcon(warningIcon);
  warningButton->setIconSize(QSize(16, 16));
  warningButton->setCheckable(true);
  warningButton->setChecked(true);
  connect(warningButton, SIGNAL(toggled(bool)), this, SLOT(handleButtonToggle(bool)));

  errorButton = new QToolButton();
  errorButton->setIcon(errorIcon);
  errorButton->setIconSize(QSize(16, 16));
  errorButton->setCheckable(true);
  errorButton->setChecked(true);
  connect(errorButton, SIGNAL(toggled(bool)), this, SLOT(handleButtonToggle(bool)));

  filterEdit = new QLineEdit();
  connect(filterEdit, SIGNAL(textChanged(const QString&)), this, SLOT(handleTextChanged(const QString&)));

  QHBoxLayout* quickFilterLayout = new QHBoxLayout();
  quickFilterLayout->addWidget(label);
  quickFilterLayout->addWidget(filterEdit, 1);
  quickFilterLayout->addWidget(traceButton);
  quickFilterLayout->addWidget(infoButton);
  quickFilterLayout->addWidget(warningButton);
  quickFilterLayout->addWidget(errorButton);
  quickFilterLayout->setMargin(0);

  QWidget* filterWidget = new QWidget();
  filterWidget->setLayout(quickFilterLayout);

  logbookWidget = new QTreeWidget();
  logbookWidget->setHeaderLabels(QStringList() << "Message" << "Module" << "UUID");
  logbookWidget->setContextMenuPolicy(Qt::ActionsContextMenu);

  QAction* action = new QAction("Set module filter", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(filterModule()));

  action = new QAction("Set UUID filter", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(filterUuid()));

  action = new QAction("Clear filter", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(clearFilter()));

  action = new QAction("Clear log", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(clearLog()));

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(filterWidget);
  mainLayout->addWidget(logbookWidget);

  setLayout(mainLayout);

  LogbookModel& model = LogbookModel::GetInstance();
  connect(&model, SIGNAL(newMessage(const std::string&, const std::string&, const std::string&, const std::string&)),
      this, SLOT(showMessage(const std::string&, const std::string&, const std::string&, const std::string&)));
}

LogbookWidget::~LogbookWidget() {
  for (int i = 0; i < logEntries.size(); ++i)
    delete logEntries[i];
}

bool LogbookWidget::matchFilter(QTreeWidgetItem* item) {
  QString filterText = filterEdit->text();

  // check if current severity
  Severity severity;
  severity = item->text(3).toAscii().data();

  switch(severity) {
  case Severity::Trace:
    if (!traceButton->isChecked())
      return false;
    break;

  case Severity::Message:
    if (!infoButton->isChecked())
      return false;
    break;

  case Severity::Warning:
    if (!warningButton->isChecked())
      return false;

  case Severity::Error:
    if (!errorButton->isChecked())
      return false;
    break;
  }

  if (filterText.length()) {
    if (item->text(0).contains(filterText) ||
        item->text(1).contains(filterText) ||
        item->text(2).contains(filterText))
    {
      return true;
    } else {
      return false;
    }
  }

  return true;
}

void LogbookWidget::filterItems() {
  logbookWidget->clear();
  QList<QTreeWidgetItem*> items;
  for (unsigned i = 0; i < logEntries.size(); ++i) {
    if (matchFilter(logEntries[i]))
      items << logEntries[i]->clone();
  }
  logbookWidget->addTopLevelItems(items);
}

void LogbookWidget::showMessage(const std::string& message, const std::string& severityStr,
    const std::string& module, const std::string& uuid)
{
  QTreeWidgetItem* item = new QTreeWidgetItem();
  Severity severity;
  severity = severityStr;
  switch(severity) {
    case Severity::Trace: item->setIcon(0, traceIcon);  break;
    case Severity::Message: item->setIcon(0, infoIcon);  break;
    case Severity::Warning: item->setIcon(0, warningIcon);  break;
    case Severity::Error: item->setIcon(0, errorIcon);  break;

    default:
      item->setIcon(0, miscIcon);
  }

  item->setText(0, message.c_str());
  item->setText(1, module.c_str());
  item->setText(2, uuid.c_str());
  item->setText(3, ((std::string)severity).c_str());

  logEntries.push_back(item);

  if (matchFilter(item)) {
    QTreeWidgetItem* newItem = item->clone();
    logbookWidget->addTopLevelItem(newItem);
    logbookWidget->scrollToItem(newItem);
  }
}

void LogbookWidget::handleButtonToggle(bool) {
  filterItems();
}

void LogbookWidget::handleTextChanged(const QString&) {
  filterItems();
}

void LogbookWidget::filterModule() {
  if (logbookWidget->currentItem())
    filterEdit->setText(logbookWidget->currentItem()->text(1));
}

void LogbookWidget::filterUuid() {
  if (logbookWidget->currentItem())
    filterEdit->setText(logbookWidget->currentItem()->text(2));
}

void LogbookWidget::clearFilter() {
  filterEdit->setText("");
}

void LogbookWidget::clearLog() {
  for (int i = 0; i < logEntries.size(); ++i)
    delete logEntries[i];
  logEntries.clear();
  logbookWidget->clear();
}

} /* namespace host */

} /* namespace gapputils */
