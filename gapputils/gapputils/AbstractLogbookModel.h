/*
 * AbstractLogbook.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_ABSTRACTLOGBOOK_H_
#define GAPPUTILS_ABSTRACTLOGBOOK_H_

#include "gapputils.h"

#include <boost/shared_ptr.hpp>
#include <sstream>

#include <capputils/Enumerators.h>

namespace gapputils {

ReflectableEnum(Severity, Trace, Message, Warning, Error);

class AbstractLogbookModel {
public:
  virtual ~AbstractLogbookModel();

  virtual void addMessage(const std::string& message,
      const Severity& severity = Severity::Message,
      const std::string& module = "<none>",
      const std::string& uuid = "<none>") = 0;
};

} /* namespace gapputils */

#endif /* ABSTRACTLOGBOOK_H_ */
