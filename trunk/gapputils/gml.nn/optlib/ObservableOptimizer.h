/**
 * @file ObservableOptimizer.h
 * @brief Common functionality of observable optimization algorithms
 *
 * @date Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_OBSERVABLEOPTIMIZER_H_
#define _OPTLIB_OBSERVABLEOPTIMIZER_H_

#include "IObservableOptimizer.h"
#include "IFunctionObserver.h"
#include "IObservableFunction.h"
#include "IFunction.h"

#include <vector>
#include <iostream>

namespace optlib {

/// Handle this event to keep track of the optimizatio progress
class OPTLIB_API ProgressEvent : public virtual IOptimizerEvent {
public:
  /// The typical three phases during the optimization
  enum Phase {
    Start,        ///< The begin of the optimization
    Iteration,    ///< Each iteration of the optimization
    End           ///< The end of the optimization
  };

public:
  Phase phase;    ///< The phase during which the event was triggered

public:
  /// Constructor to create a new ProgressEvent
  /**
   * @param[in] phase The phase during which the event was triggered
   */
  ProgressEvent(const Phase& phase) : phase(phase) { }
};

/// The ProgressEvent as a template and a new field for the current solution
template<class D>
class OPTLIB_API TProgressEvent : public ProgressEvent {
public:
  D currentSolution;    ///< The current solution

  /// Constructor to create a new TProgressEvent
  /**
   * @param[in] phase           Phasing during which this event was triggered
   * @param[in] currentSolution The currently best solution during the optimization
   */
  TProgressEvent(const Phase& phase, const D& currentSolution) : ProgressEvent(phase),
      currentSolution(currentSolution) { }
};

/// An abstract implementation of the IObservableOptimizer
/**
 * @remarks
 * - All methods of the IObservableOptimizer interface are implemented
 * - All methods of the IOptimizer interface are still pure virtual
 */
template<class D>
class ObservableOptimizer : public virtual IObservableOptimizer<D> {

private:
  std::vector<IOptimizerObserver<D>*> observers;

public:
  virtual void addObserver(IOptimizerObserver<D>& oberserver);

  /// Propagate all registered observer to a given input function if possible
  /**
   * @param[in] function The input function
   */
  template<class D2>
  void propagateObservers(IFunction<D2>& function);

  /// Propagate all registered observers to a given input optimizer of possible
  /**
   * @param[in] optimizer The input optimizer
   */
  template<class D2>
  void propagateObservers(IObservableOptimizer<D2>& optimizer);

  /// Propagate a given observer to a given optimizer if possible
  /**
   * @param[in] optimizer Optimizer that should receive to observer
   * @param[in] observer  The observer that should be added to the input optimizer
   */
  template<class D2>
  void propagateObserver(IObservableOptimizer<D2>& optimizer, IOptimizerObserver<D>& oberserver);

  virtual void fireEventTriggered(const IOptimizerEvent& event);
};

#include "ObservableOptimizer_template.cpp"

}

#endif /* _OPTLIB_OBSERVABLEOPTIMIZER_H_ */
