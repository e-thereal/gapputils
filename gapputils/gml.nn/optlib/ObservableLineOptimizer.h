/**
 * @file ObservableLineOptimizer.h
 * @brief Common functionality of observable line optimization algorithms
 *
 * @date Nov 27, 2008
 * @author Tom Brosch
 *
 * This file contains basically common events, that are usually triggered during
 * the optimization process and specific to line optimization algorithms.
 */

#ifndef _OPTLIB_OBSERVABLELINEOPTIMIZER_H_
#define _OPTLIB_OBSERVABLELINEOPTIMIZER_H_

#include "LogEvent.h"
#include "ILineOptimizer.h"
#include "ObservableOptimizer.h"

namespace optlib {

/// Common base for all line optimization events
class OPTLIB_API ILineOptimizerEvent : public virtual IOptimizerEvent {
};

/// Use this event for logging during a line optimization algorithm
class LineOptimizerLogEvent : public virtual ILineOptimizerEvent, public LogEvent {
public:
  /// Contructor to create a new log event
  /**
   * @param[in] message The loggin message
   */
  LineOptimizerLogEvent(const std::string& message) : LogEvent(message) { }
};

/// Combination of ILineOptimizer and ObservableOptimizer
class OPTLIB_API ObservableLineOptimizer : public virtual ILineOptimizer,
    public ObservableOptimizer<ILineOptimizer::DomainType>
{
public:
  /// Constructor to create a new ObservableLineOptimizer object
  ObservableLineOptimizer();
};

}


#endif /* _OPTLIB_OBSERVABLELINEOPTIMIZER_H_ */
