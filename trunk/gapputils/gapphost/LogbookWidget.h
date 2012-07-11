/*
 * LogbookWidget.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_LOGBOOKWIDGET_H_
#define GAPPUTILS_HOST_LOGBOOKWIDGET_H_

#include <qtreewidget.h>

namespace gapputils {

namespace host {

class LogbookWidget : public QTreeWidget {
  Q_OBJECT

public:
  LogbookWidget(QWidget* parent = 0);
  virtual ~LogbookWidget();

public Q_SLOTS:
  void showMessage(const std::string& message, const std::string& module);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_LOGBOOKWIDGET_H_ */
