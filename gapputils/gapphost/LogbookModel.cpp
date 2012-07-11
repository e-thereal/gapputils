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

LogbookModel& LogbookModel::GetInstance(const std::string& module) {
  LogbookModel& model = GetInstance();
  model.setModule(module);
  return model;
}

void LogbookModel::setModule(const std::string& module) {
  this->module = module;
}

void LogbookModel::addMessage(const std::string& message) {
  std::cout << module << ": " << message << std::endl;
  Q_EMIT newMessage(message, module);
}

Logbook::Logbook(const std::string& module) : module(module) { }

void Logbook::addMessage(const std::string& message) {
  LogbookModel::GetInstance(module)() << message;
}

} /* namespace host */

} /* namespace gapputils */
