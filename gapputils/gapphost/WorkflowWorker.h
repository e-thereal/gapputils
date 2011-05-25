/*
 * WorkflowWorker.h
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#ifndef WORKFLOWWORKER_H_
#define WORKFLOWWORKER_H_

#include <qthread.h>

#include "Node.h"
#include <gapputils/IProgressMonitor.h>

namespace gapputils {

namespace workflow {

class Workflow;

class WorkflowWorker : public QThread, public virtual IProgressMonitor {
  Q_OBJECT

private:
  Workflow* workflow;
  WorkflowWorker* worker;
  Node* currentNode;

public:
  WorkflowWorker(Workflow* workflow);
  virtual ~WorkflowWorker();

  virtual void run();
  virtual void reportProgress(int i);

public Q_SLOTS:
  void updateModule(workflow::Node* node);

Q_SIGNALS:
  void moduleUpdated(workflow::Node* node);
  void progressed(workflow::Node* node, int i);

};

}

}

#endif /* WORKFLOWWORKER_H_ */
