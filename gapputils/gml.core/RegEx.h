/*
 * RegEx.h
 *
 *  Created on: Feb 8, 2012
 *      Author: tombr
 */

#ifndef GML_REGEX_H_
#define GML_REGEX_H_

#include <gapputils/DefaultWorkflowElement.h>
#include <gapputils/namespaces.h>

namespace gml {

namespace core {

class RegEx : public DefaultWorkflowElement<RegEx> {

  InitReflectableClass(RegEx)

  Property(Input, std::string)
  Property(Regex, std::string)
  Property(Format, std::string)
  Property(Output, std::string)

private:
  static int inputId, regexId, formatId;

public:
  RegEx();

  void changedHandler(ObservableClass* sender, int eventId);
};

}

}

#endif /* GML_REGEX_H_ */
