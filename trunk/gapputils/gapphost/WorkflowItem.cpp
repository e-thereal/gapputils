/*
 * WorkflowItem.cpp
 *
 *  Created on: May 24, 2011
 *      Author: tombr
 */

#include "WorkflowItem.h"

#include <iostream>

namespace gapputils {

WorkflowItem::WorkflowItem(workflow::Node* node, Workbench *bench)
 : QObject(), ToolItem(node, bench)
{
}

WorkflowItem::~WorkflowItem() {
}

void WorkflowItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  workflow::Workflow* workflow = dynamic_cast<workflow::Workflow*>(getNode());
  if (workflow) {
    Q_EMIT showWorkflowRequest(workflow);
  }

  ToolItem::mouseDoubleClickEvent(event);
}

}
