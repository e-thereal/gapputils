/*
 * WorkflowItem.h
 *
 *  Created on: May 24, 2011
 *      Author: tombr
 */

#ifndef WORKFLOWITEM_H_
#define WORKFLOWITEM_H_

#include <qobject.h>
#include "ToolItem.h"
#include "Workflow.h"

namespace gapputils {

class WorkflowItem : public QObject, public ToolItem {
  Q_OBJECT

public:
  WorkflowItem(const std::string& label, Workbench *bench = 0);
  virtual ~WorkflowItem();

  virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);

Q_SIGNALS:
  // TODO: how must this request be designed? Workflow information is not present
  //void showWorkflowRequest(workflow::Workflow* workflow);
};

}

#endif /* WORKFLOWITEM_H_ */
