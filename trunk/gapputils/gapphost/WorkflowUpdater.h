#pragma once
#ifndef GAPPUTILS_HOST_WORKFLOWUPDATER_H_
#define GAPPUTILS_HOST_WORKFLOWUPDATER_H_

#include <qthread.h>

#include <gapputils/IProgressMonitor.h>
#include <boost/enable_shared_from_this.hpp>

#include <stack>

namespace gapputils {

namespace workflow {

class Node;

}

namespace host {

class WorkflowUpdater : public QThread, public virtual workflow::IProgressMonitor,
                        public boost::enable_shared_from_this<WorkflowUpdater>
{
  Q_OBJECT

private:
  boost::weak_ptr<workflow::Node> node;
  boost::weak_ptr<workflow::Node> currentNode;
  boost::weak_ptr<WorkflowUpdater> rootThread;
  bool abortRequested;

  std::stack<boost::weak_ptr<workflow::Node> > nodesStack;
  boost::shared_ptr<WorkflowUpdater> updater;

public:
  WorkflowUpdater(boost::shared_ptr<WorkflowUpdater> rootThread = boost::shared_ptr<WorkflowUpdater>());
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
  void buildStack(boost::shared_ptr<workflow::Node> node);
  void updateNodes();

public Q_SLOTS:

  // The root thread handles node updates by invoking the writeResults method
  void handleNodeUpdateFinished(boost::shared_ptr<workflow::Node> node);

  // The root thread does not only delegate the event, it also in charge of calling writeResults if requested
  void handleAndDelegateProgressedEvent(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode);

  // Simply delegate the events
  void delegateUpdateFinished();

Q_SIGNALS:
  void progressed(boost::shared_ptr<workflow::Node> node, double progress);

  /// Used internally
  void progressed(boost::shared_ptr<workflow::Node> node, double progress, bool updateNode);
  void nodeUpdateFinished(boost::shared_ptr<workflow::Node> node);
  void updateFinished();
};

}

}

#endif
