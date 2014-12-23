/**
 * @file BrentOptimizer.h
 * @brief Contains Brent's optimization algorithm
 *
 * @date Nov 4, 2008
 * @author Jon McAusland, Erick Wong, Roger Tam, Tom Brosch
 *
 * This file was mainly written by the first three authors. It was rewritten by
 * Tom Brosch to fit smoothly into the overall optimization framework.
 */

#ifndef _OPTLIB_BRENTOPTIMIZER_H_
#define _OPTLIB_BRENTOPTIMIZER_H_

#include "IParameterizable.h"
#include "ObservableLineOptimizer.h"
#include "LogEvent.h"

#include "optlib.h"

namespace optlib {

/// Stores information during an optimization step
typedef struct {
  double d, e, v, w;
  double f_v, f_w;
  double middleValue;
} BrentState;

/// Base event for all events related to Brent's method
class OPTLIB_API BrentEvent : public ILineOptimizerEvent {
public:
  BrentEvent() { }
  virtual ~BrentEvent() { }
};

/// Fired when a new main step is reached
class OPTLIB_API BrentStepChanged : public BrentEvent {
public:

  /// Basic steps during the optimization
  enum Steps {
    Start,
    FindBracket,      ///< Fired right before the bracket search starts
    Initialize,       ///< Fired right before the initialization
    Iterate,          ///< Fired before the iterative optimization starts
    Finished          ///< Fired when the algorithm has ended
  } Step;

  BrentStepChanged(Steps step) : Step(step) { }
};

/// Fired when a new main step is reached
class OPTLIB_API BrentIteration : public BrentEvent {
public:

  /// Basic steps during the optimization
  ObservableLineOptimizer::DomainType& left;
  ObservableLineOptimizer::DomainType& middle;
  ObservableLineOptimizer::DomainType& right;

  BrentIteration(ObservableLineOptimizer::DomainType& left, 
      ObservableLineOptimizer::DomainType& middle,
      ObservableLineOptimizer::DomainType& right)
   : left(left), middle(middle), right(right) { }
};

/// Implementation of Brent's method to optimize a function
/**
 * @remarks
 *
 * - This optimizer is used internally by classes implementing the
 *   IDirectionSetOptimizer interface
 *
 * - Details are taken from Richard P. Brent "Algorithms for Minimization
 *   Without Derivatives", page 79f.
 *
 * @par Known issues:
 *
 * - This implementation expects to find a minimum or maximum near by the starting point
 * - If no minimum or maximum can be found, and exception is thrown during the
 *   bracket search.
 */
class OPTLIB_API BrentOptimizer : public ObservableLineOptimizer,
                                  public virtual IParameterizable
{
private:
  double tolerance;
  double stepSize;
  bool checkMinLowerBound, checkMaxUpperBound;
  double minLowerBound, maxUpperBound;

public:
  BrentOptimizer();

  virtual std::string getName() const;

  virtual void minimize(DomainType& result, IFunction<DomainType>& function);
  virtual void maximize(DomainType& result, IFunction<DomainType>& function);

  virtual void setParameter(int id, double value);
  virtual void setParameter(int id, void* value);

  /// Sets the initial step size for the bracket search
  /**
   * @param[in] stepSize The initial step size to be used for the bracket search
   */
  void setStepSize(double stepSize);

  /// Sets the tolerance in which the solution should lie
  /**
   * @param[in] tol The tolerance.
   */
  void setTolerance(double tol);

protected:
  void findBracket(DomainType& left, DomainType& middle, DomainType& right,
      IFunction<DomainType>& function);

  /// written by Erick, left as is (nearly)
  void iterate(BrentState &st, DomainType& left, DomainType& middle, DomainType& right,
      IFunction<DomainType>& function);

  double getStepSize() const;
  double getTolerance() const;
  int getMaxIterationCount() const;
};

}

#endif /* BRENTOPTIMIZER_H_ */

