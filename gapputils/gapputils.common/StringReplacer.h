/*
 * StringReplacer.h
 *
 *  Created on: May 17, 2011
 *      Author: tombr
 */

#ifndef STRINGREPLACER_H_
#define STRINGREPLACER_H_

#include <gapputils/DefaultWorkflowElement.h>

namespace gapputils {

namespace common {

class StringReplacer : public workflow::DefaultWorkflowElement<StringReplacer> {
  InitReflectableClass(StringReplacer)

  Property(Input, std::string)
  Property(Output, std::string)
  Property(Find, std::string)
  Property(Replace, std::string)

private:
  static int inputId, findId, replaceId;
  bool resumed;

public:
  StringReplacer();
  virtual ~StringReplacer();


  void changedHandler(capputils::ObservableClass* sender, int eventId);
  virtual void writeResults();
  virtual void resume();
};

}

}

#endif /* STRINGREPLACER_H_ */
