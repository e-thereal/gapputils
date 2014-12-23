/**
 * @file OptimizerException.h
 * @brief Basic exception handling for optimization processes
 *
 * @date Nov 4, 2008
 * @author Tom Brosch
 *
 * @remarks
 * - Always throw an exception instead of writing an error message and calling exit.
 * - Leave the decision how to proceed in case of an error to the user of the API.
 */

#ifndef _OPTLIB_OPTIMIZEREXCEPTION_H_
#define _OPTLIB_OPTIMIZEREXCEPTION_H_

#include <exception>
#include <string>
#include <iostream>

#include "optlib.h"

namespace optlib {

/// Thrown when an error occurs during the optimization
class OPTLIB_API OptimizerException : public std::exception {
protected:
  std::string filename;         ///< File in which the error occurred
  int line;                     ///< Line in which the exception was thrown
  std::string description;      ///< A brief description of the error

public:

  /// Constructor to create a new OptimizationException
  /**
   * @param[in] filename    The name of the file, in which the error occurred
   * @param[in] line        The line, in which the exception was thrown
   * @param[in] description A brief description of the error
   */
  OptimizerException(const std::string& filename, int line,
      const std::string& description);

  virtual ~OptimizerException() throw();

  /// Gets the error description
  /**
   * @return A brief description of the error
   */
  std::string getDescription() const;

  /// Gets the name of the file, in which the error occurred
  /**
   * @return The name of the file, in which the error occurred
   */
  std::string getFilename() const;

  /// Gets the line in which the exception was thrown
  /**
   * @return Line in which the exception was thrown
   */
  int getLine() const;

  /// Returns a C-style character string describing the general cause of the current error.
  /**
   * @return The error description.
   */
  virtual const char* what() const throw();
};

}

/// Prints the exception to an output stream
OPTLIB_API std::ostream& operator<<(std::ostream& os, const optlib::OptimizerException& ex);

/// Use this macro to throw an exception.
/**
 * @remarks
 * - The filename and the line, in which the error occurred are automatically added
 *   to the exception context
 * - Example:
 *   @code
 *   THROW_OPTEX("Here was something going wrong");
 *   @endcode
 */
#define THROW_OPTEX(a) throw OptimizerException(__FILE__, __LINE__, a)

#endif /* _OPTLIB_OPTIMIZEREXCEPTION_H_ */
