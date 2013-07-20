/*
 * WorkflowItem.cpp
 *
 *  Created on: May 24, 2011
 *      Author: tombr
 */

#include "WorkflowItem.h"

#include <iostream>

namespace gapputils {

WorkflowItem::WorkflowItem(const std::string& label, Workbench *bench)
 : ToolItem(label, bench), doubleClicked(false)
{
}

WorkflowItem::~WorkflowItem() {
}

void WorkflowItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
  if (doubleClicked)
    Q_EMIT showWorkflowRequest(this);
  doubleClicked = false;
  ToolItem::mouseReleaseEvent(event);
}

void WorkflowItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  doubleClicked = true;
  ToolItem::mouseDoubleClickEvent(event);
}

}
