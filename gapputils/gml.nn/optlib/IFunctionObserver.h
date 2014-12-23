/**
 * @file IFunctionObserver.h
 * @brief The IFunctionObserver interface
 *
 * @date Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IFUNCTIONOBSERVER_H_
#define _OPTLIB_IFUNCTIONOBSERVER_H_

#include "IFunction.h"
#include "IObserver.h"

namespace optlib {

/// The interface of a function observer
/**
 * Implement this interface to keep track of all function evaluations during the
 * optimization process
 */

template<class T>
class OPTLIB_API IFunctionObserver : public virtual IObserver {
public:

  /// Implement this method to react on function evaluations
  /**
   * @param[in] parameter The parameter, that was evaluated
   * @param[in] result    The result of the evaluation
   * @param[in] sender    The function, that was evaluated
   */
  virtual void evaluationPerformed(const T& parameter, const double& result,
      IFunction<T>& sender) = 0;
};

}

#endif /* _OPTLIB_IFUNCTIONOBSERVER_H_ */
