/*
 * RegEx.h
 *
 *  Created on: Feb 8, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_COMMON_REGEX_H_
#define GAPPUTILS_COMMON_REGEX_H_

#include <gapputils/WorkflowElement.h>

namespace gapputils {

namespace common {

class RegEx : public gapputils::workflow::WorkflowElement {

  InitReflectableClass(RegEx)

  Property(Input, std::string)
  Property(Output, std::string)
  Property(Regex, std::string)
  Property(Format, std::string)

private:
  static int inputId, regexId, formatId;

public:
  RegEx();
  virtual ~RegEx();

  virtual void execute(gapputils::workflow::IProgressMonitor* monitor) const;
  virtual void writeResults();

  void changedHandler(capputils::ObservableClass* sender, int eventId);
};

}

}

#endif /* GAPPUTILS_COMMON_REGEX_H_ */
