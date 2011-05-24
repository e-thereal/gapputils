/*
 * StringReplacer.h
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#ifndef STRINGREPLACER_H_
#define STRINGREPLACER_H_

#include "DefaultWorkflowElement.h"

namespace gapputils {

class StringReplacer : public workflow::DefaultWorkflowElement {
  InitReflectableClass(StringReplacer)

  Property(Input, std::string)
  Property(Output, std::string)
  Property(Find, std::string)
  Property(Replace, std::string)

private:
  static int inputId, findId, replaceId;

public:
  StringReplacer();
  virtual ~StringReplacer();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

#endif /* STRINGREPLACER_H_ */