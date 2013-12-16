/*
 * GlobalPropertiesView.h
 *
 *  Created on: Jul 12, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_GLOBALPROPERTIESVIEW_H_
#define GAPPUTILS_HOST_GLOBALPROPERTIESVIEW_H_

#include <qwidget.h>
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <qtreewidget.h>

namespace gapputils {

namespace workflow {

class Workflow;

}

namespace host {

class GlobalPropertiesView : public QWidget {
  Q_OBJECT

private:
  boost::weak_ptr<workflow::Workflow> workflow;
  QTreeWidget* propertiesWidget;

public:
  GlobalPropertiesView(QWidget* parent = 0);
  virtual ~GlobalPropertiesView();

  void setWorkflow(boost::shared_ptr<workflow::Workflow> workflow);

public Q_SLOTS:
  void handleItemDoubleClicked(QTreeWidgetItem* item, int column);

Q_SIGNALS:
  void selectModuleRequested(const QString& uuid);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_GLOBALPROPERTIESVIEW_H_ */
