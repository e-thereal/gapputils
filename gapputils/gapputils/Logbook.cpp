/*
 * Logbook.cpp
 *
 *  Created on: Jul 11, 2012
 *      Author: tombr
 */

#include "Logbook.h"

#include <cassert>
#include <iostream>

namespace gapputils {

LogEntry::LogEntry(Logbook* logbook, const Severity& severity)
 : message(new std::stringstream()), logbook(logbook), severity(severity)
{
  assert(logbook);
}

LogEntry::~LogEntry() {
  logbook->addMessage(message->str(), severity);
}

Logbook::Logbook(AbstractLogbookModel* model)
 : model(model), severity(Severity::Message), module("<none>"), uuid("<none>")
{ }

Logbook::~Logbook() { }

LogEntry Logbook::operator()() {
  return LogEntry(this, severity);
}

LogEntry Logbook::operator()(const Severity& severity) {
  return LogEntry(this, severity);
}

void Logbook::setModel(AbstractLogbookModel* model) {
  this->model = model;
}

void Logbook::setSeverity(const Severity& severity) {
  this->severity = severity;
}

void Logbook::setModule(const std::string& module) {
  this->module = module;
}

void Logbook::setUuid(const std::string& uuid) {
  this->uuid = uuid;
}

void Logbook::addMessage(const std::string& message, const Severity& severity) {
  if (model) {
    model->addMessage(message, severity, module, uuid);
  } else {
    std::cout << "[" << severity << "] " << message << " (" << module << ", " << uuid << ")" << std::endl;
  }
}

} /* namespace gapputils */
