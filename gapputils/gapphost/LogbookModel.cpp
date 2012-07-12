/*
 * LogbookModel.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "LogbookModel.h"

#include <iostream>

namespace gapputils {

namespace host {

LogbookModel::LogbookModel() : QObject() {
}

LogbookModel::~LogbookModel() {
}

LogbookModel& LogbookModel::GetInstance() {
  static LogbookModel* instance = 0;
  return (instance ? *instance : *(instance = new LogbookModel()));
}

void LogbookModel::addMessage(const std::string& message, const capputils::Severity& severity,
    const std::string& module, const std::string& uuid)
{
//  std::cout << module << ": " << message << std::endl;
  Q_EMIT newMessage(message, severity, module, uuid);
}

} /* namespace host */

} /* namespace gapputils */
