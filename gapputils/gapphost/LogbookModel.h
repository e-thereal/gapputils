/*
 * LogbookModel.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_LOGBOOKMODEL_H_
#define GAPPUTILS_HOST_LOGBOOKMODEL_H_

#include <gapputils/AbstractLogbook.h>

#include <qobject.h>

namespace gapputils {

namespace host {

class LogbookModel : public QObject, public AbstractLogbook {

  Q_OBJECT

private:
  std::string module;

protected:
  LogbookModel();

public:
  virtual ~LogbookModel();

  static LogbookModel& GetInstance();
  static LogbookModel& GetInstance(const std::string& module);

  void setModule(const std::string& module);

  virtual void addMessage(const std::string& message);

Q_SIGNALS:
  void newMessage(const std::string& message, const std::string& module);
};

class Logbook : public AbstractLogbook {
private:
  std::string module;

public:
  Logbook(const std::string& module);
  virtual void addMessage(const std::string& message);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_LOGBOOKMODEL_H_ */
