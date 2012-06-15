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
  bool abortRequested;

public:
  WorkflowWorker(Workflow* workflow);
  virtual ~WorkflowWorker();

  virtual void run();
  virtual void reportProgress(double progress, bool updateNode = false);
  virtual bool getAbortRequested() const;

  void abort();

public Q_SLOTS:
  void updateModule(workflow::Node* node, bool force);

Q_SIGNALS:
  void moduleUpdated(workflow::Node* node);
  void progressed(workflow::Node* node, double progress, bool updateNode = false);

};

}

}

#endif /* WORKFLOWWORKER_H_ */
