/**
 * @file IOptimizerObserver.h
 * @brief The IOptimizerObserver interface
 *
 * @date   Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IOPTIMIZEROBSERVER_H_
#define _OPTLIB_IOPTIMIZEROBSERVER_H_

#include "IObserver.h"
#include "IOptimizer.h"

#include "IOptimizerEvent.h"

namespace optlib {

/// Tells, that a class can handle optimization events
/**
 * Implement this interface to keep track of all optimization events during the
 * optimization process
 */
template<class D>
class OPTLIB_API IOptimizerObserver : public virtual IObserver {

public:
  /// Implement this method to handle optimization events
  /**
   * @param[in] event   The event that has been triggered
   * @param[in] sender  The optimization algorithm, which fired the event
   */
  virtual void eventTriggered(const IOptimizerEvent& event, IOptimizer<D>& sender) = 0;
};

}

#endif /* _OPTLIB_IOPTIMIZEROBSERVER_H_ */
