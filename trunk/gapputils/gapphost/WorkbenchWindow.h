/*
 * WorkbenchWindow.h
 *
 *  Created on: Aug 3, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_WORKBENCHWINDOW_H_
#define GAPPUTILS_HOST_WORKBENCHWINDOW_H_

#include <qmdisubwindow.h>

#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

namespace gapputils {

class Workbench;

namespace workflow {
  class Workflow;
  class Node;
  class Edge;
}

namespace host {

class WorkbenchWindow : public QMdiSubWindow {

  Q_OBJECT

private:
  boost::weak_ptr<workflow::Workflow> workflow;
  Workbench* workbench;

public:
  WorkbenchWindow(boost::shared_ptr<workflow::Workflow> workflow, QWidget* parent = 0);
  virtual ~WorkbenchWindow();

  void createItem(boost::shared_ptr<workflow::Node> node);
  bool createCable(boost::shared_ptr<workflow::Edge> edge);

  void addNodesFromClipboard();

protected:
  void closeEvent(QCloseEvent *event);

private Q_SLOTS:
  void createModule(int x, int y, QString classname);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_WORKBENCHWINDOW_H_ */
