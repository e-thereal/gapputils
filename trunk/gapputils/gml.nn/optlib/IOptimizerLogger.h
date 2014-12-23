/**
 * @file IOptimizerLogger.h
 * @brief The IOptimizerLogger interface
 *
 * @date   Dec 17, 2009
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IOPTIMIZERLOGGER_H_
#define _OPTLIB_IOPTIMIZERLOGGER_H_

#include "optlib.h"

#include <iostream>

namespace optlib {

/// Tells that a class logs optimization progress
/**
 * OptimizerObserver which are used for logging should implement this interface
 */
class OPTLIB_API IOptimizerLogger {

public:
  /// Prints the log to an output stream
  /**
   * @param[in] out (Optional) The output stream. If not set std::cout is used
   */
  virtual void print(std::ostream& out = std::cout) const = 0;
};

}

/// Prints the log to an output stream
OPTLIB_API std::ostream& operator<<(std::ostream& os, const optlib::IOptimizerLogger& logger);

#endif /* _OPTLIB_IOPTIMIZERLOGGER_H_ */
