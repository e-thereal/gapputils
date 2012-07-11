/*
 * AbstractLogbook.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "AbstractLogbook.h"

#include <cassert>

namespace gapputils {

LogEntry::LogEntry(AbstractLogbook* logbook) : message(new std::stringstream()), logbook(logbook) {
  assert(logbook);
}

LogEntry::~LogEntry() {
  logbook->addMessage(message->str());
}

AbstractLogbook::AbstractLogbook() {
}

AbstractLogbook::~AbstractLogbook() {
}

LogEntry AbstractLogbook::operator()() {
  return LogEntry(this);
}

} /* namespace gapputils */
