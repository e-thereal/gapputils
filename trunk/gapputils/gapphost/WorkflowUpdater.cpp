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
  : rootThread(rootThread), abortRequested(false)
{
  if (!rootThread) {
    updater = boost::shared_ptr<WorkflowUpdater>(new WorkflowUpdater(this));

    // One workflow updater instance is executed in the root thread
    // this instance can be used to synchronize the root thread with the updater threads
    // Therefore, some signals send from the updater threads are delegated by the root thread instance
    // Others are handled by the root thread directly
    connect(updater.get(), SIGNAL(progressed(boost::shared_ptr<workflow::Node>, double, bool)), this, SLOT(handleAndDelegateProgressedEvent(boost::shared_ptr<workflow::Node>, double, bool)), Qt::BlockingQueuedConnection);
    connect(updater.get(), SIGNAL(nodeUpdateFinished(boost::shared_ptr<workflow::Node>)), this, SLOT(handleNodeUpdateFinished(boost::shared_ptr<workflow::Node>)), Qt::BlockingQueuedConnection);
    connect(updater.get(), SIGNAL(finished()), this, SLOT(delegateUpdateFinished()));
  }
}

WorkflowUpdater::~WorkflowUpdater(void) { }

void WorkflowUpdater::update(boost::shared_ptr<workflow::Node> node) {
  abortRequested = false;
  if (!rootThread) {
    capputils::Logbook dlog(&LogbookModel::GetInstance());
    dlog.setModule("gapputils::host::WorkflowUpdater");
    dlog() << "Workflow update started.";
    updater->update(node);
  } else {
    this->node = node;
    start();
  }
}

void WorkflowUpdater::run() {
  // TODO: implement the three update methods here
  //       1. Combiner case: workflow + CollectionElement interface nodes
  //       2. Workflow case: workflow
  //       3. Single node case: else

  capputils::Logbook dlog(&LogbookModel::GetInstance());
  dlog.setModule("gapputils::host::WorkflowUpdater");
  dlog.setSeverity(capputils::Severity::Trace);

  ChecksumUpdater checksumUpdater;
  boost::shared_ptr<workflow::Workflow> workflow = boost::dynamic_pointer_cast<workflow::Workflow>(node.lock());
  if (workflow) {
    if (workflow->hasCollectionElementInterface()) {

      dlog() << "Combiner case";

      /*** Combiner case ***/

      // Update input (this is important to know how many elements need to be processed
      // reset collection
      // while advance all collections
      // add all interface nodes to the stack
      // and update nodes
      // append results at the end

      std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = workflow->getInterfaceNodes();
      std::vector<boost::shared_ptr<workflow::CollectionElement> > collectionElements;
      std::set<boost::shared_ptr<workflow::CollectionElement> > inputElements;

      bool needsUpdate = true;
      bool lastIteration = false;

      checksumUpdater.update(node.lock());
      for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
        if (!workflow->isOutputNode(interfaceNodes[i].lock()))
          buildStack(interfaceNodes[i].lock());
      }
      updateNodes();

      for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
        boost::shared_ptr<workflow::CollectionElement> collection = boost::dynamic_pointer_cast<workflow::CollectionElement>(interfaceNodes[i].lock()->getModule());
        if (collection && collection->getCalculateCombinations()) {
          if (!collection->resetCombinations()) {
            dlog(capputils::Severity::Warning) << "Can't update workflow. Empty or null input collection detected.";
            needsUpdate = false;
          }
          collectionElements.push_back(collection);
          collection->setCalculateCombinations(false);
          if (!workflow->isOutputNode(interfaceNodes[i].lock())) {
            inputElements.insert(collection);
            if (collection->getCurrentIteration() + 1 == collection->getIterationCount()) {
              lastIteration = true;
              dlog() << "Last iteration";
            }
          }
        }
      }

      dlog() << "#Elements: " << collectionElements.size();
      dlog() << "#Inputs: " << inputElements.size();

      while (needsUpdate) {
        
        if (lastIteration) {
          dlog() << "Resetting combiner flag";
          for (unsigned i = 0; i < collectionElements.size(); ++i)
            collectionElements[i]->setCalculateCombinations(true);
        }

        // build stacks
        checksumUpdater.update(node.lock());
        for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
          if (workflow->isOutputNode(interfaceNodes[i].lock()))
            buildStack(interfaceNodes[i].lock());
        }

        // update
        updateNodes();

        // write all results first before advance collection in order to avoid side-effects due to
        // automatic updates triggered by the advance step
        for (unsigned i = 0; i < collectionElements.size(); ++i)
          collectionElements[i]->appendResults();

        // advance collections
        for (unsigned i = 0; i < collectionElements.size(); ++i) {
          const bool isInput = inputElements.find(collectionElements[i]) != inputElements.end();
          if (!collectionElements[i]->advanceCombinations() && isInput)
            needsUpdate = false;

          if (isInput && collectionElements[i]->getCurrentIteration() + 1 == collectionElements[i]->getIterationCount())
            lastIteration = true;
        }
      }

      // The combiner flag should be reset in all cases.
      // Resetting is necessary if not a single update cycle was performed
      for (unsigned i = 0; i < collectionElements.size(); ++i)
        collectionElements[i]->setCalculateCombinations(true);

    } else {

      /*** Workflow case ***/

      dlog() << "Workflow case";

      checksumUpdater.update(node.lock());
      std::vector<boost::weak_ptr<workflow::Node> >& interfaceNodes = workflow->getInterfaceNodes();
      for (unsigned i = 0; i < interfaceNodes.size(); ++i) {
        if (workflow->isOutputNode(interfaceNodes[i].lock()))
          buildStack(interfaceNodes[i].lock());
      }
      updateNodes();
    }
  } else {

    /*** Single update case ***/

    checksumUpdater.update(node.lock());
    buildStack(node.lock());
    updateNodes();
  }
}

void WorkflowUpdater::reportProgress(double progress, bool updateNode) {
  if (currentNode)
    reportProgress(currentNode, progress, updateNode);
}

void WorkflowUpdater::reportProgress(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode) {
  Q_EMIT progressed(node, progress, updateNode);
}

void WorkflowUpdater::reportNodeUpdateFinished(boost::shared_ptr<workflow::Node> node) {
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

void WorkflowUpdater::buildStack(boost::shared_ptr<workflow::Node> node) {
  capputils::Logbook dlog(&LogbookModel::GetInstance());

  // Rebuild the stack without node, thus guaranteeing that node appears only once
  std::stack<boost::weak_ptr<workflow::Node> > oldStack;
  while (!nodesStack.empty()) {
    oldStack.push(nodesStack.top());
    nodesStack.pop();
  }
  while (!oldStack.empty()) {
    boost::weak_ptr<workflow::Node> n = oldStack.top();
    if (n.lock() != node)
      nodesStack.push(n);
    oldStack.pop();
  }

  // Only add it if it needs an update or if it is the first node
  if (node->getInputChecksum() != node->getOutputChecksum() || nodesStack.empty()) {
    reportProgress(node, 0, false);
    nodesStack.push(node);
    //dlog() << "Added node to build stack: " << node->getUuid();
  } else {
    reportProgress(node, 100, false);
  }
  
  // call build stack for all dependent nodes
  std::vector<boost::shared_ptr<workflow::Node> > dependentNodes;
  node->getDependentNodes(dependentNodes);
  for (unsigned i = 0; i < dependentNodes.size(); ++i)
    buildStack(dependentNodes[i]);
}

void WorkflowUpdater::updateNodes() {
  capputils::Logbook dlog(&LogbookModel::GetInstance());
  
  // Go through the stack and update all nodes
  while(!nodesStack.empty() && !getAbortRequested()) {
    currentNode = nodesStack.top().lock();
    nodesStack.pop();

    reportProgress(currentNode, ToolItem::InProgress, false);

    // update node
    // TODO: check if value could be read from cache. Never read it from cache if it is the last node
    if (nodesStack.empty() || currentNode->getInputChecksum() != currentNode->getOutputChecksum()) {
      boost::shared_ptr<workflow::Workflow> workflow = boost::dynamic_pointer_cast<workflow::Workflow>(currentNode);
      if (workflow) {

        // Create a new worker for the sub workflow
        WorkflowUpdater updater(rootThread);

        // Just sit and watch if he is doing a good job
        connect(&updater, SIGNAL(progressed(boost::shared_ptr<workflow::Node>, double, bool)), rootThread, SLOT(handleAndDelegateProgressedEvent(boost::shared_ptr<workflow::Node>, double, bool)), Qt::BlockingQueuedConnection);
        connect(&updater, SIGNAL(nodeUpdateFinished(boost::shared_ptr<workflow::Node>)), rootThread, SLOT(handleNodeUpdateFinished(boost::shared_ptr<workflow::Node>)), Qt::BlockingQueuedConnection);

        dlog.setModule(currentNode->getModule()->getClassName());
        dlog.setUuid(currentNode->getUuid());
        dlog(capputils::Severity::Trace) << "Starting update. (" << currentNode->getInputChecksum() << ", " << currentNode->getOutputChecksum() << ")";

        // Let him get started
        updater.update(currentNode);

        // And chill until the work is done
        updater.wait();

      } else if (nodesStack.empty() || !NodeCache::Restore(currentNode)) {
        boost::shared_ptr<workflow::WorkflowElement> element = boost::dynamic_pointer_cast<workflow::WorkflowElement>(currentNode->getModule());
        if (element) {
          dlog.setModule(element->getClassName());
          dlog.setUuid(currentNode->getUuid());
          dlog(capputils::Severity::Trace) << "Starting update. (" << currentNode->getInputChecksum() << ", " << currentNode->getOutputChecksum() << ")";
          element->execute(this);
        }
      }
    }

    // update finished
    if (!getAbortRequested()) {
      reportProgress(currentNode, 100.0, false);
      reportNodeUpdateFinished(currentNode);
    }

    this->currentNode.reset();
  }
}

// The root thread handles node updates by invoking the writeResults method
void WorkflowUpdater::handleNodeUpdateFinished(boost::shared_ptr<workflow::Node> node) {
  node->setOutputChecksum(node->getInputChecksum());
  boost::shared_ptr<workflow::WorkflowElement> element = boost::dynamic_pointer_cast<workflow::WorkflowElement>(node->getModule());
  if (element) {
    element->writeResults();
  }

  // TODO: Avoid update if state was read from cache
  NodeCache::Update(node);
}

// The root thread does not only delegate the event, it also in charge of calling writeResults if requested
void WorkflowUpdater::handleAndDelegateProgressedEvent(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode) {
  if (updateNode) {
    boost::shared_ptr<workflow::WorkflowElement> element = boost::dynamic_pointer_cast<workflow::WorkflowElement>(node->getModule());
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
