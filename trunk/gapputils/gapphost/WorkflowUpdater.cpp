#include "WorkflowUpdater.h"

#include "Node.h"
#include "Workflow.h"
#include "ToolItem.h"
#include "NodeCache.h"

#include <gapputils/WorkflowElement.h>
#include <gapputils/CollectionElement.h>

#include <cassert>
#include <iostream>

#include "ChecksumUpdater.h"

#include <capputils/Logbook.h>
#include "LogbookModel.h"

namespace gapputils {

namespace host {

WorkflowUpdater::WorkflowUpdater(WorkflowUpdater* rootThread)
  : node(0), currentNode(0), rootThread(rootThread), abortRequested(false), updater(0)
{
  if (!rootThread) {
    updater = new WorkflowUpdater(this);

    // One workflow updater instance is executed in the root thread
    // this instance can be used to synchronize the root thread with the updater threads
    // Therefore, some signals send from the updater threads are delegated by the root thread instance
    // Others are handled by the root thread directly
    connect(updater, SIGNAL(progressed(workflow::Node*, double, bool)), this, SLOT(handleAndDelegateProgressedEvent(workflow::Node*, double, bool)), Qt::BlockingQueuedConnection);
    connect(updater, SIGNAL(nodeUpdateFinished(workflow::Node*)), this, SLOT(handleNodeUpdateFinished(workflow::Node*)), Qt::BlockingQueuedConnection);
    connect(updater, SIGNAL(finished()), this, SLOT(delegateUpdateFinished()));
  }
}

WorkflowUpdater::~WorkflowUpdater(void)
{
  if (!rootThread) {
    assert(updater);
    delete updater;
    updater = 0;
  } else {
    assert(!updater);
  }
}

void WorkflowUpdater::update(workflow::Node* node) {
  abortRequested = false;
  if (!rootThread) {
    updater->update(node);
  } else {
    capputils::Logbook dlog(&LogbookModel::GetInstance());
    dlog.setModule("gapputils::host::WorkflowUpdater");
    dlog() << "Workflow update started.";
    this->node = node;
    start();
  }
}

void WorkflowUpdater::run() {
  // TODO: implement the three update methods here
  //       1. Combiner case: workflow + CollectionElement interface nodes
  //       2. Workflow case: workflow
  //       3. Single node case: else

  ChecksumUpdater checksumUpdater;
  workflow::Workflow* workflow = dynamic_cast<workflow::Workflow*>(node);
  if (workflow) {
    if (workflow->hasCollectionElementInterface()) {

      /*** Combiner case ***/

      // reset collection
      // while advance all collections
      // add all interface nodes to the stack
      // and update nodes
      // append results at the end
      std::vector<workflow::Node*>& interfaceNodes = workflow->getInterfaceNodes();
      std::vector<workflow::CollectionElement*> collectionElements;

      bool needsUpdate = true;
      bool lastIteration = false;

      for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
        workflow::CollectionElement* collection = dynamic_cast<workflow::CollectionElement*>(interfaceNodes[i]->getModule());
        if (collection && collection->getCalculateCombinations()) {
          if (!collection->resetCombinations())
            needsUpdate = false;
          collectionElements.push_back(collection);
          collection->setCalculateCombinations(false);
          if (collection->getCurrentIteration() + 1 == collection->getIterationCount())
            lastIteration = true;
        }
      }

      while (needsUpdate) {
        
        if (lastIteration) {
          for (unsigned i = 0; i < collectionElements.size(); ++i)
            collectionElements[i]->setCalculateCombinations(true);
        }

        // build stacks
        checksumUpdater.update(node);
        for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
          if (workflow->isOutputNode(interfaceNodes[i]))
            buildStack(interfaceNodes[i]);
        }

        // update
        updateNodes();

        // write all results first before advance collection in order to avoid side-effects due to
        // automatic updates triggered by the advance step
        for (unsigned i = 0; i < collectionElements.size(); ++i)
          collectionElements[i]->appendResults();

        // advance collections
        for (unsigned i = 0; i < collectionElements.size(); ++i) {
          if (!collectionElements[i]->advanceCombinations())
            needsUpdate = false;
          if (collectionElements[i]->getCurrentIteration() + 1 == collectionElements[i]->getIterationCount())
            lastIteration = true;
        }
      }
    } else {

      /*** Workflow case ***/

      checksumUpdater.update(node);
      std::vector<workflow::Node*>& interfaceNodes = workflow->getInterfaceNodes();
      for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
        if (workflow->isOutputNode(interfaceNodes[i]))
          buildStack(interfaceNodes[i]);
      }
      updateNodes();
    }
  } else {

    /*** Single update case ***/

    checksumUpdater.update(node);
    buildStack(node);
    updateNodes();
  }
}

void WorkflowUpdater::reportProgress(double progress, bool updateNode) {
  if (currentNode)
    reportProgress(currentNode, progress, updateNode);
}

void WorkflowUpdater::reportProgress(workflow::Node* node, double progress, bool updateNode) {
  Q_EMIT progressed(node, progress, updateNode);
}

void WorkflowUpdater::reportNodeUpdateFinished(workflow::Node* node) {
  Q_EMIT nodeUpdateFinished(node);
}

bool WorkflowUpdater::getAbortRequested() const {
  if (rootThread)
    return rootThread->getAbortRequested();
  return abortRequested;
}

void WorkflowUpdater::abort() {
  abortRequested = true;
}

void WorkflowUpdater::buildStack(workflow::Node* node) {

  // Rebuild the stack without node, thus guaranteeing that node appears only once
  std::stack<workflow::Node*> oldStack;
  while (!nodesStack.empty()) {
    oldStack.push(nodesStack.top());
    nodesStack.pop();
  }
  while (!oldStack.empty()) {
    workflow::Node* n = oldStack.top();
    if (n != node)
      nodesStack.push(n);
    oldStack.pop();
  }

  // Only add it if it needs an update or if it is the first node
  if (node->getInputChecksum() != node->getOutputChecksum() || nodesStack.empty()) {
    reportProgress(node, 0, false);
    nodesStack.push(node);
  } else {
    reportProgress(node, 100, false);
  }
  
  // call build stack for all dependent nodes
  std::vector<workflow::Node*> dependentNodes;
  node->getDependentNodes(dependentNodes);
  for (unsigned i = 0; i < dependentNodes.size(); ++i)
    buildStack(dependentNodes[i]);
}

void WorkflowUpdater::updateNodes() {
  
  // Go through the stack and update all nodes
  while(!nodesStack.empty() && !getAbortRequested()) {
    currentNode = nodesStack.top();
    nodesStack.pop();

    reportProgress(currentNode, ToolItem::InProgress, false);

    workflow::Workflow* workflow = dynamic_cast<workflow::Workflow*>(currentNode);
    if (workflow) {

      // Create a new worker for the sub workflow
      WorkflowUpdater updater(rootThread);

      // Just sit and watch if he is doing a good job
      connect(&updater, SIGNAL(progressed(workflow::Node*, double, bool)), rootThread, SLOT(handleAndDelegateProgressedEvent(workflow::Node*, double, bool)), Qt::BlockingQueuedConnection);
      connect(&updater, SIGNAL(nodeUpdateFinished(workflow::Node*)), rootThread, SLOT(handleNodeUpdateFinished(workflow::Node*)), Qt::BlockingQueuedConnection);

      // Let him get started
      updater.update(currentNode);

      // And chill until the work is done
      updater.wait();

    } else {
      // update node
      // TODO: check if value could be read from cache. Never read it from cache if it is the last node
      if (nodesStack.empty() || (currentNode->getInputChecksum() != currentNode->getOutputChecksum()
          && !NodeCache::Restore(currentNode)))
      {
        workflow::WorkflowElement* element = dynamic_cast<workflow::WorkflowElement*>(currentNode->getModule());
        if (element) {
          element->execute(this);
        }
      }
    }

    // update finished
    if (!getAbortRequested()) {
      reportProgress(currentNode, 100.0, false);
      reportNodeUpdateFinished(currentNode);
    }

    currentNode = 0;
  }
}

// The root thread handles node updates by invoking the writeResults method
void WorkflowUpdater::handleNodeUpdateFinished(workflow::Node* node) {
  node->setOutputChecksum(node->getInputChecksum());
  workflow::WorkflowElement* element = dynamic_cast<workflow::WorkflowElement*>(node->getModule());
  if (element) {
    element->writeResults();
  }

  // TODO: Avoid update if state was read from cache
  NodeCache::Update(node);
}

// The root thread does not only delegate the event, it also in charge of calling writeResults if requested
void WorkflowUpdater::handleAndDelegateProgressedEvent(workflow::Node* node, double progress, bool updateNode) {
  if (updateNode) {
    workflow::WorkflowElement* element = dynamic_cast<workflow::WorkflowElement*>(node->getModule());
    if (element) {
      element->writeResults();
    }
  }
  Q_EMIT progressed(node, progress);
}

void WorkflowUpdater::delegateUpdateFinished() {
  capputils::Logbook dlog(&LogbookModel::GetInstance());
  dlog.setModule("gapputils::host::WorkflowUpdater");
  dlog() << "Workflow update finished.";
  Q_EMIT updateFinished();
}

}

}
