/*
 * WorkflowToolBox.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_WORKFLOWTOOLBOX_H_
#define GAPPUTILS_HOST_WORKFLOWTOOLBOX_H_

#include <qwidget.h>
#include <qlineedit.h>
#include <qtreewidget.h>

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace host {

class WorkflowToolBox : public QWidget {

  Q_OBJECT

private:
  QLineEdit* toolBoxFilterEdit;
  QTreeWidget* toolBox;
  std::map<QTreeWidgetItem*, boost::shared_ptr<std::vector<QTreeWidgetItem* > > > toolBoxItems;

protected:
  WorkflowToolBox(QWidget * parent = 0);

public:
  virtual ~WorkflowToolBox();

  static WorkflowToolBox& GetInstance() {
    static WorkflowToolBox* instance = 0;
    return (instance ? *instance : *(instance = new WorkflowToolBox()));
  }

  void update();

Q_SIGNALS:
  void itemSelected(QString classname);

public Q_SLOTS:
  void focusFilter();
  void filterToolBox(const QString& text);
  void itemClickedHandler(QTreeWidgetItem *item, int column);
  void currentItemChangedHandler(QTreeWidgetItem* current, QTreeWidgetItem* previous);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_WORKFLOWTOOLBOX_H_ */
