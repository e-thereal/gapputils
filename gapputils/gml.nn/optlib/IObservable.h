/**
 * @file IObservable.h
 * @brief The IObservable interface
 *
 * @date Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IOBSERVABLE_H_
#define _OPTLIB_IOBSERVABLE_H_

#include "IObserver.h"

namespace optlib {

/// Tells, that a class can be observed.
class OPTLIB_API IObservable {
public:

  /// Adds an observer to the optimization algorithm
  /**
   * @param[in] observer The observer to be added.
   */
  virtual void addObserver(IObserver& observer) = 0;
};

}

#endif /* _OPTLIB_IOBSERVABLE_H_ */
