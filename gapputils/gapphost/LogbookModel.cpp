/*
 * LogbookModel.cpp
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#include "LogbookModel.h"
#include "DataModel.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace gapputils {

namespace host {

LogbookModel::LogbookModel() : QObject() {
  fs::path path(DataModel::getInstance().getLogfileName());
  if (!path.parent_path().empty())
    fs::create_directories(path.parent_path());

  logname = DataModel::getInstance().getLogfileName();
  logfile.open(logname.c_str());
  struct tm* timeinfo;
  char buffer[256];
  time_t currentTime = time(0);

  timeinfo = localtime(&currentTime);
  strftime(buffer, 256, "%H:%M:%S", timeinfo);

  logfile << "Time stamp\tSeverity\tMessage\tModule\tUUID" << std::endl;
  logfile << "[" << buffer << "]\tMessage\tStarting log session" << std::endl;
  logfile.flush();
}

LogbookModel::~LogbookModel() {
  logfile << "End of log session." << std::endl;
  logfile.close();
}

LogbookModel& LogbookModel::GetInstance() {
  static LogbookModel* instance = 0;
  return (instance ? *instance : *(instance = new LogbookModel()));
}

void LogbookModel::addMessage(const std::string& message, const capputils::Severity& severity,
    const std::string& module, const std::string& uuid)
{
  struct tm* timeinfo;
  char buffer[256];
  time_t currentTime = time(0);

  timeinfo = localtime(&currentTime);
  strftime(buffer, 256, "%H:%M:%S", timeinfo);

  std::string currentlogname = DataModel::getInstance().getLogfileName();
  if (currentlogname != logname) {
    logname = currentlogname;
    logfile.close();

    fs::path path(DataModel::getInstance().getLogfileName());
    fs::create_directories(path.parent_path());

    logfile.open(logname.c_str());
    logfile << "Time stamp\tSeverity\tMessage\tModule\tUUID" << std::endl;
    logfile << "[" << buffer << "]\tMessage\tStarting log session" << std::endl;
  }

  logfile << "[" << buffer << "]\t" << severity << "\t" << message << "\t" << module << "\t" << uuid << std::endl;
  logfile.flush();

  Q_EMIT newMessage(message, severity, module, uuid);
}

} /* namespace host */

} /* namespace gapputils */
