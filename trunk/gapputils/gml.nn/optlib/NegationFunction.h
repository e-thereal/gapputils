/**
 * @file NegationFunction.h
 * @brief This functions always negates the return value of the input function
 *
 * @date Nov 3, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_NEGATIONFUNCTION_H_
#define _OPTLIB_NEGATIONFUNCTION_H_

#include "IParameterizedFunction.h"

namespace optlib {

/// This functions always negates the return value of the input function
/**
 * @remarks
 * - You can use this function to implement the minimize() method of the IOptimizer
 *   interface only, and implement the maximize() method by calculating the minimum
 *   of the negated function.
 */
template<class D>
class OPTLIB_API NegationFunction : public virtual IParameterizedFunction<D> {
protected:
  IFunction<D>* function;             ///< The input function

public:

  /// Constructor that creates a new NegationFunction for a given input function
  /**
   * @param[in] function The input function
   */
  NegationFunction(IFunction<D>* function) : function(function) { }

  /// Evaluates the input function and returns the negated return value
  /**
   * @param[in] parameter The function parameter of DomainType
   *
   * @return The negated function value of the input function
   */
  virtual double eval(const D& parameter) {
    return -function->eval(parameter);
  }

  virtual void setParameter(int id, double value) {
    IParameterizedFunction<D>* paraFunc;
    if ((paraFunc = dynamic_cast<IParameterizedFunction<D>* >(function)))
      paraFunc->setParameter(id, value);
  }

  virtual void setParameter(int id, void* value) {
    IParameterizedFunction<D>* paraFunc;
    if ((paraFunc = dynamic_cast<IParameterizedFunction<D>* >(function)))
      paraFunc->setParameter(id, value);
  }
};

}

#endif /* _OPTLIB_NEGATIONFUNCTION_H_ */
