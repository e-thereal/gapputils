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
#include <qsettings.h>

#include <ctime>
#include <iostream>

namespace gapputils {
namespace host {

enum TableColumns {TimeColumn, MessageColumn, ModuleColumn, UuidColumn, SeverityColumn};

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
  logbookWidget->setHeaderLabels(QStringList() << "Time" << "Message" << "Module" << "UUID");
  logbookWidget->setContextMenuPolicy(Qt::ActionsContextMenu);
  logbookWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);

  QAction* action = new QAction("Filter module", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(filterModule()));

  action = new QAction("Filter UUID", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(filterUuid()));

  action = new QAction("Clear filter", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(clearFilter()));

  action = new QAction(this);
  action->setSeparator(true);
  logbookWidget->addAction(action);

  action = new QAction("Clear log", this);
  logbookWidget->addAction(action);
  connect(action, SIGNAL(triggered()), this, SLOT(clearLog()));

  connect(logbookWidget, SIGNAL(itemDoubleClicked(QTreeWidgetItem*, int)),
      this, SLOT(handleItemDoubleClicked(QTreeWidgetItem*, int)));

  QVBoxLayout* mainLayout = new QVBoxLayout();
  mainLayout->addWidget(filterWidget);
  mainLayout->addWidget(logbookWidget);

  setLayout(mainLayout);

  LogbookModel& model = LogbookModel::GetInstance();
  connect(&model, SIGNAL(newMessage(const std::string&, const std::string&, const std::string&, const std::string&)),
      this, SLOT(showMessage(const std::string&, const std::string&, const std::string&, const std::string&)));

  QSettings settings;
  if (settings.contains("logbook/TimeWidth"))
    logbookWidget->setColumnWidth(TimeColumn, settings.value("logbook/TimeWidth").toInt());
  if (settings.contains("logbook/MessageWidth"))
    logbookWidget->setColumnWidth(MessageColumn, settings.value("logbook/MessageWidth").toInt());
  if (settings.contains("logbook/ModuleWidth"))
    logbookWidget->setColumnWidth(ModuleColumn, settings.value("logbook/ModuleWidth").toInt());
  if (settings.contains("logbook/UuidWidth"))
    logbookWidget->setColumnWidth(UuidColumn, settings.value("logbook/UuidWidth").toInt());
}

LogbookWidget::~LogbookWidget() {
  QSettings settings;
  settings.setValue("logbook/TimeWidth", logbookWidget->columnWidth(TimeColumn));
  settings.setValue("logbook/MessageWidth", logbookWidget->columnWidth(MessageColumn));
  settings.setValue("logbook/ModuleWidth", logbookWidget->columnWidth(ModuleColumn));
  settings.setValue("logbook/UuidWidth", logbookWidget->columnWidth(UuidColumn));

  for (int i = 0; i < logEntries.size(); ++i)
    delete logEntries[i];
}

bool LogbookWidget::matchFilter(QTreeWidgetItem* item) {
  QString filterText = filterEdit->text();

  // check if current severity
  Severity severity;
  severity = item->text(SeverityColumn).toAscii().data();

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
    if (item->text(MessageColumn).contains(filterText) ||
        item->text(ModuleColumn).contains(filterText) ||
        item->text(UuidColumn).contains(filterText))
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

  struct tm* timeinfo;
  char buffer[256];
  time_t currentTime = time(0);

  timeinfo = localtime(&currentTime);
  strftime(buffer, 256, "%H:%M:%S", timeinfo);

  item->setText(TimeColumn, buffer);
  item->setText(MessageColumn, message.c_str());
  item->setText(ModuleColumn, module.c_str());
  item->setText(UuidColumn, uuid.c_str());
  item->setText(SeverityColumn, ((std::string)severity).c_str());

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
    filterEdit->setText(logbookWidget->currentItem()->text(ModuleColumn));
}

void LogbookWidget::filterUuid() {
  if (logbookWidget->currentItem())
    filterEdit->setText(logbookWidget->currentItem()->text(UuidColumn));
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

void LogbookWidget::handleItemDoubleClicked(QTreeWidgetItem* item, int column) {
  Q_EMIT selectModuleRequested(item->text(UuidColumn));
}

} /* namespace host */

} /* namespace gapputils */
