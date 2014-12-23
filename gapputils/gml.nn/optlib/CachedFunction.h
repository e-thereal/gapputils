/**
 * @file CachedFunction.h
 * @brief Implementation of the function cache
 *
 * @date   Nov 24, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_CACHEDFUNCTION_H_
#define _OPTLIB_CACHEDFUNCTION_H_

#include "IParameterizedFunction.h"

#include <map>
#include <iostream>

namespace optlib {

/// Turns an arbitrary function into a cached function
/**
 * @remarks
 *
 * - Being a cached function is somehow like a feature. (See MultistepOptimizer)
 * - To add a cache to a function, use the following:
 * @code
 *   CachedFunction<MyFunction> myCachedFunction;
 * @endcode
 *
 * - So far, the base class must implement the IParameterizedFunction interface,
 *   because setParameter delegates its call to its base class. This should be
 *   changed in future versions.
 *
 * - You wonder, whether it is a good idea to combine this feature with the
 *   ObservableFunction feature? Sure, it is! That's what features were made for. Usually,
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
class CachedFunction : public F, public virtual IParameterizedFunction<typename F::DomainType> {
private:
  std::map<typename F::DomainType, double> valueCache;

public:
  virtual double eval(const typename F::DomainType& parameter);

  /// Sets a parameter
  /**
   * @remarks
   * - The cache is cleared automatically, whenever a parameter has been changed
   *
   * @param[in] id    The unique ID of the parameter
   * @param[in] value The parameter value
   */
  virtual void setParameter(int id, double value);
  virtual void setParameter(int id, void* value);
};

#include "CachedFunction_template.cpp"

}

#endif /* _OPTLIB_CACHEDFUNCTION_H_ */
