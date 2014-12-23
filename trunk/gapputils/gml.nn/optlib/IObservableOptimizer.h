/**
 * @file IObservableOptimizer.h
 * @brief The IObservableOptimizer interface
 *
 * @date   Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IOBSERVABLEOPTIMIZER_H_
#define _OPTLIB_IOBSERVABLEOPTIMIZER_H_

#include "IOptimizer.h"
#include "IOptimizerObserver.h"

namespace optlib {

/// Tells, that it is possible to keep track of the progress of an optimization algorithm
template<class D>
class OPTLIB_API IObservableOptimizer : public virtual IOptimizer<D> {
public:
  typedef IOptimizerObserver<D> ObserverType;   ///< The type of the observer

public:
  /// Adds an observer to the optimization algorithm
  /**
   * @param[in] observer The optimization observer to be added
   */
  virtual void addObserver(ObserverType& observer) = 0;

  /// Fires an optimization event (for internal use only)
  virtual void fireEventTriggered(const IOptimizerEvent& event) = 0;
};

}

#endif /* _OPTLIB_IOBSERVABLEOPTIMIZER_H_ */
