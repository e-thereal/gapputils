/**
 * @file ObservableFunction.h
 * @brief Contains the implementation the IObservableFunction interface as a feature
 *
 * @date Nov 5, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_OBSERVABLEFUNCTION_H_
#define _OPTLIB_OBSERVABLEFUNCTION_H_

#include "IObservableFunction.h"

#include <vector>

namespace optlib {

/// Turns an arbitrary function into an observable function
/**
 * @remarks
 *
 * - Being an observable function is somehow like a feature. (See MultistepOptimizer)
 * - In order to make a function observable, use the following
 * @code
 *   ObservableFunction<MyFunction> myObservableFunction;
 * @endcode
 *
 * - You wonder, whether it is a good idea to combine this feature with caching
 *   (CachedFunction)? Sure, it is! That's what features were made for. Usually,
 *   you have to consider the right order, but since these two features are completely
 *   independent of each other, it doesn't matter at all. So writing:
 *   @code
 *   CachedFunction<ObservableFunction<MyFunction> > myObservableCachedFunction;
 *   @endcode
 *   has the same effect as writing:
 *   @code
 *   ObservableFunction<CachedFunction<MyFunction> > myObservableCachedFunction;
 *   @endcode
 */
template<class F>
class ObservableFunction : public F, public virtual IObservableFunction<typename F::DomainType> {
private:
  /// Vector to store all observers
  std::vector<IFunctionObserver<typename F::DomainType>* > observers;

public:
  virtual void addObserver(IFunctionObserver<typename F::DomainType>& oberserver);

  /// Overload this method to define the function
  /**
   * @param[in] parameter The function parameter of DomainType
   *
   * @return The function value of RangeType
   *
   * @remarks
   * - Overwritten to notify that a function evaluation has been performed
   */
  virtual double eval(const typename F::DomainType& parameter);

protected:

  /// Fires the an evaluation performed event
  /**
   * @remarks
   * - This should be a real event in feature versions
   */
  void fireEvaluationPerformed(const typename F::DomainType& parameter, const double& result);
};

#include "ObservableFunction_template.cpp"

}

#endif /* _OPTLIB_OBSERVABLEFUNCTION_H_ */
