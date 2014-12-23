/**
 * @file IObservableFunction.h
 * @brief The IObservableFunction interface
 *
 * @date Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IOBSERVABLEFUNCTION_H_
#define _OPTLIB_IOBSERVABLEFUNCTION_H_

#include "IFunction.h"
#include "IFunctionObserver.h"

namespace optlib {

/// Tells, that the function can be observed.
template<class T>
class OPTLIB_API IObservableFunction : public virtual IFunction<T> {
public:

  /// Adds an observer to the function
  /**
   * @param[in] observer The function observer to be added
   */
  virtual void addObserver(IFunctionObserver<T>& observer) = 0;
};

}

#endif /* _OPTLIB_IOBSERVABLEFUNCTION_H_ */
