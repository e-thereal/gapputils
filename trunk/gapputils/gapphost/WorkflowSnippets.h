/*
 * WorkflowSnippets.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_WORKFLOWSNIPPETS_H_
#define GAPPUTILS_HOST_WORKFLOWSNIPPETS_H_

#include <qwidget.h>
#include <qlineedit.h>
#include <qtreewidget.h>

#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace host {

class WorkflowSnippets : public QWidget {

  Q_OBJECT

private:
  QLineEdit* toolBoxFilterEdit;
  QTreeWidget* toolBox;
  std::map<QTreeWidgetItem*, boost::shared_ptr<std::vector<QTreeWidgetItem* > > > toolBoxItems;

protected:
  WorkflowSnippets(QWidget * parent = 0);

public:
  virtual ~WorkflowSnippets();

  static WorkflowSnippets& GetInstance() {
    static WorkflowSnippets* instance = 0;
    return (instance ? *instance : *(instance = new WorkflowSnippets()));
  }

  void update();

public Q_SLOTS:
  void focusFilter();
  void filterToolBox(const QString& text);
  void itemClickedHandler(QTreeWidgetItem *item, int column);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_WORKFLOWSNIPPETS_H_ */
