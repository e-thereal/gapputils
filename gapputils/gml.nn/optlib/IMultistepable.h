/**
 * @file IMultistepable.h
 * @brief The IMultistepable interface
 *
 * @date Dec 11, 2008
 * @author: Tom Brosch
 */

#ifndef _OPTLIB_IMULTISTEPABLE_H_
#define _OPTLIB_IMULTISTEPABLE_H_

namespace optlib {

/// Tells, that the optimization is done in more than one step.
class IMultistepable {
public:

  /// Call this function to start a new parameter set
  /**
   * @remarks
   * - Calling this function will create a new set of parameters and therefore,
   *   the overall optimization will take one more step
   */
  virtual void newParameterSet() = 0;

  /// Adds a new parameter to the current parameter set
  /**
   * @param[in] id    The unique ID of the parameter
   * @param[in] value (Optional) The parameter value (defaults to zero for improper properties)
   */
  virtual void addParameter(int id, double value = 0.0) = 0;

  /// Adds a new parameter to the current parameter set
  /**
   * @param[in] id    The unique ID of the parameter
   * @param[in] value Pointer to an object containing the parameter
   */
  virtual void addParameter(int id, void* value) = 0;
};

}

#endif /* _OPTLIB_IMULTISTEPABLE_H_ */
