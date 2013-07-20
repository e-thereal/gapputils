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

class WorkflowItem : public ToolItem {
  Q_OBJECT

protected:
  bool doubleClicked;

public:
  WorkflowItem(const std::string& label, Workbench *bench = 0);
  virtual ~WorkflowItem();

  virtual void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
  virtual void mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event);

Q_SIGNALS:
  void showWorkflowRequest(ToolItem* item);
};

}

#endif /* WORKFLOWITEM_H_ */
