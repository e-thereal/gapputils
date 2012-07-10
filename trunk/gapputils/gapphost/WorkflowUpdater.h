#pragma once
#ifndef GAPPUTILS_HOST_WORKFLOWUPDATER_H_
#define GAPPUTILS_HOST_WORKFLOWUPDATER_H_

#include <qthread.h>

#include <gapputils/IProgressMonitor.h>

#include <stack>

namespace gapputils {

namespace workflow {

class Node;

}

namespace host {

class WorkflowUpdater : public QThread, public virtual workflow::IProgressMonitor
{
  Q_OBJECT

private:
  workflow::Node* node;
  workflow::Node* currentNode;
  WorkflowUpdater* rootThread;
  bool abortRequested, collectionInterfaceMode;

  std::stack<workflow::Node*> nodesStack;
  WorkflowUpdater* updater;

public:
  WorkflowUpdater(WorkflowUpdater* rootThread = 0);
  virtual ~WorkflowUpdater(void);

  void update(workflow::Node* node);
  virtual void run();

  // From IProgressMonitor
  virtual void reportProgress(double progress, bool updateNode = false);
  void reportProgress(workflow::Node* node, double progress, bool updateNode = false);
  void reportNodeUpdateFinished(workflow::Node* node);
  virtual bool getAbortRequested() const;
  void abort();

private:
  void buildStack(workflow::Node* node);
  void updateNodes();

public Q_SLOTS:

  // The root thread handles node updates by invoking the writeResults method
  void handleNodeUpdateFinished(workflow::Node* node);

  // The root thread does not only delegate the event, it also in charge of calling writeResults if requested
  void handleAndDelegateProgressedEvent(workflow::Node* node, double progress, bool updateNode);

  // Simply delegate the events
  void delegateUpdateFinished();

Q_SIGNALS:
  void progressed(workflow::Node* node, double progress, bool updateNode);
  void nodeUpdateFinished(workflow::Node* node);
  void updateFinished();
};

}

}

#endif