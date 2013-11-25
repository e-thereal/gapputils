#pragma once
#ifndef GAPPUTILS_HOST_WORKFLOWUPDATER_H_
#define GAPPUTILS_HOST_WORKFLOWUPDATER_H_

#include <qthread.h>

#include <gapputils/IProgressMonitor.h>
#include <boost/weak_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <vector>
#include <stack>
#include <set>

namespace gapputils {

namespace workflow {

class Node;
class CollectionElement;

}

namespace host {

class WorkflowUpdater : public QThread, public virtual workflow::IProgressMonitor
{
  Q_OBJECT

private:
  boost::weak_ptr<workflow::Node> node;
  boost::shared_ptr<workflow::Node> currentNode;
  WorkflowUpdater* rootThread;
  bool abortRequested;
  bool needsUpdate;
  bool lastIteration;

  std::vector<boost::weak_ptr<workflow::Node> > interfaceNodes;
  std::vector<boost::shared_ptr<workflow::CollectionElement> > collectionElements;
  std::set<boost::shared_ptr<workflow::CollectionElement> > inputElements;

  std::stack<boost::weak_ptr<workflow::Node> > nodesStack;
  boost::shared_ptr<WorkflowUpdater> updater;

public:
  WorkflowUpdater(WorkflowUpdater* rootThread = NULL);
  virtual ~WorkflowUpdater(void);

  void update(boost::shared_ptr<workflow::Node> node);
  virtual void run();

  // From IProgressMonitor
  virtual void reportProgress(double progress, bool updateNode = false);
  void reportProgress(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode = false);
  void reportNodeUpdateFinished(boost::shared_ptr<workflow::Node> node);
  virtual bool getAbortRequested() const;
  void abort();

private:
  void initializeCollectionLoop();
  void resetCollectionFlag();
  void advanceCollectionLoop();

  void buildStack(boost::shared_ptr<workflow::Node> node);
  void updateNodes();
  void resetNode(boost::shared_ptr<workflow::Node> node);

public Q_SLOTS:

  // The root thread handles node updates by invoking the writeResults method
  void handleNodeUpdateFinished(boost::shared_ptr<workflow::Node> node);

  // The root thread does not only delegate the event, it also in charge of calling writeResults if requested
  void handleAndDelegateProgressedEvent(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode);

  // Simply delegate the events
  void delegateUpdateFinished();

  void initializeCollectionLoop(WorkflowUpdater* updater);
  void resetCollectionFlag(WorkflowUpdater* updater);
  void advanceCollectionLoop(WorkflowUpdater* updater);

Q_SIGNALS:
  void progressed(boost::shared_ptr<workflow::Node> node, double progress);

  /// Used internally
  void progressed(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode);
  void nodeUpdateFinished(boost::shared_ptr<workflow::Node> node);
  void updateFinished();

  void initializeCollectionLoopRequested(WorkflowUpdater* updater);
  void resetCollectionFlagRequested(WorkflowUpdater* updater);
  void advanceCollectionLoopRequested(WorkflowUpdater* updater);
};

}

}

#endif
