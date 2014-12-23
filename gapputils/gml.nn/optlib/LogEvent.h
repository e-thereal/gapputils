/**
 * @file LogEvent.h
 * @brief Basic logging features using events
 *
 * @date Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_LOGEVENT_H_
#define _OPTLIB_LOGEVENT_H_

#include "IOptimizerEvent.h"

#include <string>
#include <iostream>

namespace optlib {

/// Basic event used for logging
/**
 * @remarks
 * - Always use this event instead of writing to the output or error stream. Leave
 *   the decision, how to print logging messages, to the UI programmer
 * - You can get the same behavior as writing to the output stream by handling this
 *   event and printing this event to the output stream
 */
class OPTLIB_API LogEvent : public virtual IOptimizerEvent {
protected:
  const std::string& message;                   ///< The log message

public:
  /// Contructor to create a new log event
  /**
   * @param[in] message The loggin message
   */
  LogEvent(const std::string& message);

  /// Get the message of the log event
  /**
   * @return The log message
   */
  const std::string& getMessage() const;
};

}

/// Prints an LogEvent to an output stream
std::ostream& operator<<(std::ostream& os, const optlib::LogEvent& event);

#endif /* _OPTLIB_LOGEVENT_H_ */
