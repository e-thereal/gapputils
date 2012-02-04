/*
 * WorkflowWorker.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowWorker.h"

#include "Workflow.h"
#include <gapputils/WorkflowElement.h>

#include <iostream>

using namespace std;

namespace gapputils {

namespace workflow {

WorkflowWorker::WorkflowWorker(Workflow* workflow) : QThread(), workflow(workflow), worker(0) {
  //cout << "[" << QThread::currentThreadId() << "] " << "Worker created." << endl;
}

WorkflowWorker::~WorkflowWorker() {
  if (worker) {
    disconnect(worker, SIGNAL(progressed(workflow::Node*, int)), workflow, SLOT(showProgress(workflow::Node*, int)));
    disconnect(worker, SIGNAL(moduleUpdated(workflow::Node*)), workflow, SLOT(finalizeModuleUpdate(workflow::Node*)));
    disconnect(workflow, SIGNAL(processModule(workflow::Node*, bool)), worker, SLOT(updateModule(workflow::Node*, bool)));
    delete worker;
  }
}

void WorkflowWorker::run() {
  //cout << "[" << QThread::currentThreadId() << "] " << "start thread." << endl;

  worker = new WorkflowWorker(0);
  connect(worker, SIGNAL(progressed(workflow::Node*, int)), workflow, SLOT(showProgress(workflow::Node*, int)));
  connect(worker, SIGNAL(moduleUpdated(workflow::Node*)), workflow, SLOT(finalizeModuleUpdate(workflow::Node*)));
  connect(workflow, SIGNAL(processModule(workflow::Node*, bool)), worker, SLOT(updateModule(workflow::Node*, bool)));

  exec();
  //cout << "[" << QThread::currentThreadId() << "] " << "thread finished" << endl;
}

void WorkflowWorker::updateModule(Node* node, bool force) {
  currentNode = node;
  node->update(this, force);

  Q_EMIT moduleUpdated(node);
}

void WorkflowWorker::reportProgress(int i) {
  Q_EMIT progressed(currentNode, i);
}

}

}
