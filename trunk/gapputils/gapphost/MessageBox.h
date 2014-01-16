/*
 * MessageBox.h
 *
 *  Created on: Jan 16, 2014
 *      Author: tombr
 */

#ifndef GAPPHOST_MESSAGEBOX_H_
#define GAPPHOST_MESSAGEBOX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace interfaces {

class MessageBox : public DefaultWorkflowElement<MessageBox> {

  InitReflectableClass(MessageBox)

  Property(WriteToLogbook, bool)

private:
  bool resumed;

public:
  MessageBox();

  void handleChanged(ObservableClass* sender, int eventId);
  virtual void resume();
};

} /* namespace interfaces */

#endif /* GAPPHOST_MESSAGEBOX_H_ */
