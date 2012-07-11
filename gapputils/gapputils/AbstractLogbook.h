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

namespace gapputils {

class AbstractLogbook;

class LogEntry {
private:
  boost::shared_ptr<std::stringstream> message;
  AbstractLogbook* logbook;

public:
  LogEntry(AbstractLogbook* logbook);
  virtual ~LogEntry();

  template<class T>
  std::ostream& operator<<(const T& value) {
    return *message << value;
  }
};

class AbstractLogbook {
public:
  AbstractLogbook();
  virtual ~AbstractLogbook();

  LogEntry operator()();

  virtual void addMessage(const std::string& message) = 0;
};



} /* namespace gapputils */

#endif /* ABSTRACTLOGBOOK_H_ */
