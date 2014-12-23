/**
 * @file IFunction.h
 * @brief The IFunction interface
 *
 * @date Nov 3, 2008
 * @author Tom Brosch
 */

#ifndef _OPTLIB_IFUNCTION_H_
#define _OPTLIB_IFUNCTION_H_

#include "../alglib/TypeSystem.h"
#include "optlib.h"

namespace optlib {

/// A real-valued function which defines our optimization problem.
/**
 *  In the context of optimization we consider real-valued functions only
 *
 *  @remarks
 *  - We want to use IFunction as an interface, hence always inherit
 *    from IFunction as a virtual base class.
 *
 *    Example:
 *
 *    \code
 *      class MyFunction : public virtual IFunction { ... };
 *    \endcode
 *
 *    This prevents ambiguities that might occur due to multiple inheritance!
 */
template<class T>
class OPTLIB_API IFunction : public virtual alglib::IFunction<T, double> {
};

}

#endif /* IFUNCTION_H_ */
