/*
 * LogEvent.cpp
 *
 *  Created on: Nov 5, 2008
 *      Author: tombr
 */

#include "LogEvent.h"

using namespace optlib;
using namespace std;

LogEvent::LogEvent(const string& message) : message(message) {

}

const string& LogEvent::getMessage() const {
  return message;
}

ostream& operator<<(ostream& os, const LogEvent& event) {
  os << event.getMessage();
  return os;
}
