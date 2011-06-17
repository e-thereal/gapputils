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
 : QObject(), ToolItem(label, bench)
{
}

WorkflowItem::~WorkflowItem() {
}

void WorkflowItem::mouseDoubleClickEvent(QGraphicsSceneMouseEvent* event) {
  // TODO: emit alternative signal
//  workflow::Workflow* workflow = dynamic_cast<workflow::Workflow*>(getNode());
//  if (workflow) {
//    Q_EMIT showWorkflowRequest(workflow);
//  }

  ToolItem::mouseDoubleClickEvent(event);
}

}
