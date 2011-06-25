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
 : ToolItem(label, bench)
{
}

WorkflowItem::~WorkflowItem() {
}

void WorkflowItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  Q_EMIT showWorkflowRequest(this);

  ToolItem::mouseDoubleClickEvent(event);
}

}
