/*
 * MessageBox.cpp
 *
 *  Created on: Jan 16, 2014
 *      Author: tombr
 */

#include "MessageBox.h"

#include <capputils/EventHandler.h>

namespace interfaces {

BeginPropertyDefinitions(MessageBox)

  ReflectableBase(DefaultWorkflowElement<MessageBox>)

  WorkflowProperty(WriteToLogbook, Flag(), Description("If set, changes of the labels will be written to the logbook."))

EndPropertyDefinitions

MessageBox::MessageBox() : _WriteToLogbook(false), resumed(false) {
  setLabel("Enter your message here.");

  Changed.connect(capputils::EventHandler<MessageBox>(this, &MessageBox::handleChanged));
}

void MessageBox::resume() {
  resumed = true;
}

void MessageBox::handleChanged(ObservableClass* /*sender*/, int eventId) {
  if (!resumed)
    return;

  Logbook& dlog = getLogbook();
  if (eventId == DefaultWorkflowElement<MessageBox>::labelId && getWriteToLogbook()) {
    dlog(Severity::Message) << getLabel();
  }
}

} /* namespace interfaces */
