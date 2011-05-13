/*
 * WorkflowWorker.cpp
 *
 *  Created on: May 11, 2011
 *      Author: tombr
 */

#include "WorkflowWorker.h"

#include "Workflow.h"
#include <WorkflowElement.h>

#include <iostream>

using namespace std;

namespace gapputils {

namespace workflow {

WorkflowWorker::WorkflowWorker(Workflow* workflow) : QThread(), workflow(workflow), worker(0) {
  //cout << "[" << QThread::currentThreadId() << "] " << "Worker created." << endl;
}

WorkflowWorker::~WorkflowWorker() {
  if (worker) {
    disconnect(worker, SIGNAL(progressed(Node*, int)), workflow, SLOT(showProgress(Node*, int)));
    disconnect(worker, SIGNAL(moduleUpdated(Node*)), workflow, SLOT(finalizeModuleUpdate(Node*)));
    disconnect(workflow, SIGNAL(processModule(Node*)), worker, SLOT(updateModule(Node*)));
    delete worker;
  }
}

void WorkflowWorker::run() {
  //cout << "[" << QThread::currentThreadId() << "] " << "start thread." << endl;

  worker = new WorkflowWorker(0);
  connect(worker, SIGNAL(progressed(Node*, int)), workflow, SLOT(showProgress(Node*, int)));
  connect(worker, SIGNAL(moduleUpdated(Node*)), workflow, SLOT(finalizeModuleUpdate(Node*)));
  connect(workflow, SIGNAL(processModule(Node*)), worker, SLOT(updateModule(Node*)));

  exec();
  //cout << "[" << QThread::currentThreadId() << "] " << "thread finished" << endl;
}

void WorkflowWorker::updateModule(Node* node) {
  currentNode = node;
  WorkflowElement* element = dynamic_cast<WorkflowElement*>(node->getModule());
  if (element) {
    element->execute(this);
  }

  Q_EMIT moduleUpdated(node);
}

void WorkflowWorker::reportProgress(int i) {
  Q_EMIT progressed(currentNode, i);
}

}

}
