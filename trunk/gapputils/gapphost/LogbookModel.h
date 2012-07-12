/*
 * LogbookModel.h
 *
 *  Created on: Jul 10, 2012
 *      Author: tombr
 */

#ifndef GAPPUTILS_HOST_LOGBOOKMODEL_H_
#define GAPPUTILS_HOST_LOGBOOKMODEL_H_

#include <gapputils/AbstractLogbookModel.h>

#include <qobject.h>

namespace gapputils {

namespace host {

class LogbookModel : public QObject, public AbstractLogbookModel {

  Q_OBJECT

protected:
  LogbookModel();

public:
  virtual ~LogbookModel();

  static LogbookModel& GetInstance();

  virtual void addMessage(const std::string& message,
        const Severity& severity = Severity::Message,
        const std::string& module = "<none>",
        const std::string& uuid = "<none>");

Q_SIGNALS:
  void newMessage(const std::string& message, const std::string& severity, const std::string& module, const std::string& uuid);
};

} /* namespace host */

} /* namespace gapputils */

#endif /* GAPPUTILS_HOST_LOGBOOKMODEL_H_ */
