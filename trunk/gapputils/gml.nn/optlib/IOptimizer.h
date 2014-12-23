/**
 * @file IOptimizer.h
 * @brief Basic optimizer interface
 *
 * @date Nov 3, 2008
 * @author  Tom Brosch
 */

#ifndef _OPTLIB_IOPTIMIZER_H_
#define _OPTLIB_IOPTIMIZER_H_

#include "IFunction.h"

#include <string>

/// Includes everything that has to do with optimization
/**
 * Provides all kind of classes to define optimization problems, to solve them and
 * to keep track of the progress during optimization.
 */
namespace optlib {

/// Common base for all optimization algorithms
class OPTLIB_API IOptimizerBase {
public:
  /// Virtual destructor
  virtual ~IOptimizerBase() { }

  /// Gets the name of the optimization algorithm
  virtual std::string getName() const = 0;
};

/// Base optimization algorithm interface
/**
 * Defines, that every optimization algorithm, should be able to find the minimum
 * or maximum of a given function.
 */
template<class T>
class OPTLIB_API IOptimizer : public virtual IOptimizerBase {
public:
   typedef T DomainType;

public:
  /// Find the minium of a given function
  /**
   * @throws OptimizerException
   *
   * @param[in,out] result   Initial solution and final result of the optimization
   * @param[in]     function The function to be optimized
   */
  virtual void minimize(T& result, IFunction<T>& function) = 0;

  /// Find the maximum of a given function
  /**
   * @param[in,out] result   Initial solution and final result of the optimization
   * @param[in]     function The function to be optimized
   */
  virtual void maximize(T& result, IFunction<T>& function) = 0;

};

}

#endif /* IOPTIMIZER_H_ */
