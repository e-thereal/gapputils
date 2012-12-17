/*
 * HeadlessApp.h
 *
 *  Created on: Dec 17, 2012
 *      Author: tombr
 */

#ifndef HEADLESSAPP_H_
#define HEADLESSAPP_H_

#include <qobject.h>

#include <string>
#include <map>

#include <boost/shared_ptr.hpp>

namespace gapputils {

namespace workflow {
  class Node;
}

namespace host {

class WorkflowUpdater;

class HeadlessApp : public QObject {

  Q_OBJECT

private:
  boost::shared_ptr<WorkflowUpdater> workflowUpdater;

public:
  HeadlessApp(QWidget *parent = 0);
  virtual ~HeadlessApp();

  void resume();

public Q_SLOTS:
  void updateMainWorkflow();
  void updateMainWorkflowNode(const std::string& label);
  void updateFinished();
  void showProgress(boost::shared_ptr<workflow::Node>, double);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* HEADLESSAPP_H_ */
