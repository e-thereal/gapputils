/*
 * LogbookWidget.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_LOGBOOKWIDGET_H_
#define GAPPUTILS_HOST_LOGBOOKWIDGET_H_

#include <qtreewidget.h>
#include <qicon.h>
#include <qtoolbutton.h>
#include <qlineedit.h>

#include <vector>

namespace capputils {
class Severity;
}

namespace gapputils {

namespace host {

class LogbookWidget : public QWidget {
  Q_OBJECT

private:
  QIcon traceIcon, infoIcon, warningIcon, errorIcon, miscIcon;
  QTreeWidget* logbookWidget;
  QToolButton *infoButton, *traceButton, *warningButton, *errorButton;
  QLineEdit* filterEdit;
  std::vector<QTreeWidgetItem*> logEntries;

public:
  LogbookWidget(QWidget* parent = 0);
  virtual ~LogbookWidget();

  bool matchFilter(QTreeWidgetItem* item);
  void filterItems();

public Q_SLOTS:
  void showMessage(const std::string& message, const std::string& severity, const std::string& module, const std::string& uuid);
  void handleButtonToggle(bool);
  void handleTextChanged(const QString&);
  void filterModule();
  void filterUuid();
  void clearFilter();
  void clearLog();
  void handleItemDoubleClicked(QTreeWidgetItem* item, int column);

Q_SIGNALS:
  void selectModuleRequested(const QString& uuid);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_LOGBOOKWIDGET_H_ */
