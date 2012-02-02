/*
 * StringStack.h
 *
 *  Created on: Jan 30, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_COMMON_STRINGSTACK_H_
#define GAPPUTILS_COMMON_STRINGSTACK_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class StringStack : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(StringStack)

  Property(InputVector1, std::vector<std::string>)
  Property(InputVector2, std::vector<std::string>)
  Property(OutputVector, std::vector<std::string>)

private:
  mutable StringStack* data;

public:
  StringStack();
  virtual ~StringStack();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_COMMON_STRINGSTACK_H_ */
